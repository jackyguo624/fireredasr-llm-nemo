# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union

import lightning.pytorch as pl
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import update_num_microbatches
from omegaconf.omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm.data.data_sampler import SpeechLMDataSampler
from fireredasr_llm.collections.speechllm.data.dataset.audio_text_lhotse_dataset import MultimodalConversationDataset
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging


class AudioToTextDataModule(pl.LightningDataModule, IOMixin):
    """
    Data module for speech-to-text LLM.
    """

    def __init__(self, config: Union[DictConfig, Dict], tokenizer: TokenizerSpec):
        super().__init__()
        self.cfg = OmegaConf.create(config) if not isinstance(config, DictConfig) else config
        self.tokenizer = tokenizer
        self._train_ds = None
        self._validation_ds = None
        self._test_ds = None
        self._validation_names = None
        self._test_names = None
        self.init_global_step = 0
        self.text_processor = None
        self._num_validation_dl = None
        self._num_test_dl = None

    @property
    def global_batch_size(self):
        """
        get the global batch size
        """
        return self.data_cfg.global_batch_size

    @property
    def micro_batch_size(self):
        """
        get the micro batch size
        """
        return self.data_cfg.micro_batch_size

    @property
    def seq_length(self):
        """
        get the max sequence length
        """
        return self.data_cfg.max_seq_length

    @property
    def data_cfg(self):
        """
        get the common data configuration
        """
        if 'common' not in self.cfg:
            raise ValueError("`common` configuration is missing in the data config")
        return self.cfg.common


    def prepare_data(self):
        """
        download, IO, etc. Useful with shared filesystems
        only called on 1 GPU/TPU in distributed
        """
        pass

    def setup(self, stage=None):
        """
        build datasets, text processor and data sampler
        """
        # make assignments here (train/val/test split)
        # called on every process in DDP
        if stage == 'fit' or stage is None:
            self._train_ds = self._create_dataset('train')
            self._validation_ds = self._create_dataset('validation')
        elif stage == 'validate' or stage is None:
            self._validation_ds = self._create_dataset('validation')
        if stage == 'test' or stage is None:
            self._test_ds = self._create_dataset('test')

        self.data_sampler = SpeechLMDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.data_cfg.get("rampup_batch_size", None),
            # dataloader_type="batch",  # "batch" should be used for SFT,
        )

        # Follows the calculation in `nemo.collections.nlp.data.language_modeling.megatron.
        # base_dataset_utils.get_datasets_weights_and_num_samples`
        self.max_train_samples = int(math.ceil(self.global_batch_size * self.trainer.max_steps * 1.005))

    @lru_cache
    def _create_dataset(self, mode: str):
        """
        Create the datasets for each of train/validation/test/predict mode
        """
        data_cfg = self.cfg.get(f"{mode}_ds", None)
        if data_cfg is None:
            logging.info(f"Skipping {mode} dataset creation as it is not specified in the config: {self.cfg}")
            return None

        logging.info(f"Creating Lhotse dataset for {mode}")
        if mode != 'train':
            setattr(self, f"_{mode}_names", self._parse_lhotse_data_name(mode))

        return MultimodalConversationDataset(
            tokenizer=self.tokenizer,
            tokens_to_generate=data_cfg.get('tokens_to_generate', 0),
            max_seq_length=data_cfg["max_seq_length"],
            is_train=(mode == 'train'),
        )

    def _parse_lhotse_data_name(self, mode: str) -> List[str]:
        """
        Get the dataset names for lhotse datasets
        """
        if mode == 'train':
            return []
        data_cfg = self.cfg.get(f"{mode}_ds", None)
        if data_cfg.get('manifest_filepath', None):
            manifest_filepath = data_cfg.manifest_filepath
            if isinstance(manifest_filepath, str):
                manifest_filepath = [manifest_filepath]
        elif data_cfg.get('cuts_path', None):
            manifest_filepath = data_cfg.cuts_path
            if isinstance(manifest_filepath, str):
                manifest_filepath = [manifest_filepath]
        else:
            input_cfg = data_cfg.input_cfg
            if isinstance(input_cfg, (str, Path)):
                # Resolve /path/to/input_cfg.yaml into config contents if needed.
                input_cfg = OmegaConf.load(input_cfg)
                assert len(input_cfg) == 1, "Only one dataset with multiple manifest paths is supported for eval"
                data_cfg.input_cfg = input_cfg
                # for getting names
                manifest_filepath = [ic.manifest_filepath for ic in input_cfg[0].input_cfg]

        names = []
        for cur_manifest_filepath in manifest_filepath:
            names.append(Path(cur_manifest_filepath).stem)
        logging.info(f"Parsed names for lhotse {mode} dataset: {names}")
        return names

    def _create_lhotse_dataloader(self, dataset: Any, mode: str, **kwargs) -> DataLoader:
        """
        Get lhotse dataloader
        """
        data_cfg = self.cfg.get(f"{mode}_ds", None)
        if data_cfg is None:
            logging.info(f"Skipping {mode} dataloader creation as it is not specified in the config: {self.cfg}")
            return None


        if mode == "train":
            return get_lhotse_dataloader_from_config(
                data_cfg,
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
                dataset=dataset,
            )
        # for eval, we need to create separate dataset so as to report metrics separately
        else:
            dls = []
            if data_cfg.get('manifest_filepath', None):
                manifest_filepath = data_cfg.manifest_filepath
                for cur_manifest_filepath in manifest_filepath:
                    conf = copy.deepcopy(data_cfg)
                    conf['manifest_filepath'] = cur_manifest_filepath
                    dls.append(
                        get_lhotse_dataloader_from_config(
                            conf,
                            global_rank=parallel_state.get_data_parallel_rank(),
                            world_size=parallel_state.get_data_parallel_world_size(),
                            dataset=dataset,
                        )
                    )
            elif data_cfg.get('cuts_path', None):
                cuts_path = data_cfg.cuts_path
                for cur_cuts_path in cuts_path:
                    conf = copy.deepcopy(data_cfg)
                    conf['cuts_path'] = cur_cuts_path
                    dls.append(
                        get_lhotse_dataloader_from_config(
                            conf,
                            global_rank=parallel_state.get_data_parallel_rank(),
                            world_size=parallel_state.get_data_parallel_world_size(),
                            dataset=dataset,
                        )
                    )
            else:
                input_cfg = data_cfg.input_cfg
                if isinstance(input_cfg, (str, Path)):
                    # Resolve /path/to/input_cfg.yaml into config contents if needed.
                    input_cfg = OmegaConf.load(input_cfg)
                    assert len(input_cfg) == 1, "Only one dataset with multiple manifest paths is supported for eval"
                    data_cfg.input_cfg = input_cfg
                for cur_input_cfg in input_cfg[0].input_cfg:
                    conf = copy.deepcopy(data_cfg)
                    conf.input_cfg[0].input_cfg = [cur_input_cfg]
                    dls.append(
                        get_lhotse_dataloader_from_config(
                            conf,
                            global_rank=parallel_state.get_data_parallel_rank(),
                            world_size=parallel_state.get_data_parallel_world_size(),
                            dataset=dataset,
                        )
                    )

            return dls

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Return train dataloader
        """
        self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step
        return self._create_lhotse_dataloader(self._train_ds, 'train')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Return val dataloader
        """
        data_loaders = self._create_lhotse_dataloader(self._validation_ds, 'validation')
        self._num_validation_dl = len(data_loaders) if isinstance(data_loaders, list) else 1
        return data_loaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Return test dataloader
        """
        data_loaders = self._create_lhotse_dataloader(self._test_ds, 'test')
        self._num_test_dl = len(data_loaders) if isinstance(data_loaders, list) else 1
        return data_loaders

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """
        Return predict dataloader
        """
        if "predict_ds" not in self.cfg and "test_ds" in self.cfg:
            data_key = 'test'
        elif "predict_ds" not in self.cfg and "validation_ds" in self.cfg:
            data_key = 'validation'
        else:
            data_key = 'predict'

        self._test_ds = self._create_dataset(data_key)
        return self._create_lhotse_dataloader(self._test_ds, 'predict')

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        consumed_samples = self.data_sampler.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        return {'consumed_samples': consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule stat

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """

        consumed_samples = state_dict['consumed_samples']
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples

        update_num_microbatches(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        self.data_sampler.if_first_step = 1
