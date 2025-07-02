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

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq

import nemo.collections.asr as nemo_asr
from nemo.collections.speechlm.utils import get_nested_attr, to_dict_config
from nemo.core.classes.common import Serialization
from nemo.core.classes.module import NeuralModule
from nemo.lightning import io
from nemo.utils import logging, model_utils


class MCoreASRModule(MegatronModule):
    """
    Wrapper class for ASR encoder from `nemo.collections.asr.models.ASRModel`.

    `TransformerConfig` is a dummy config to satisfy the `MegatronModule` constructor.
    `num_attention_heads` is set to 16 such that it's divisible by the value of TP.
    `num_layers` and `hidden_size` are set to 1 since not used.
    """

    def __init__(
        self,
        encoder: NeuralModule,
        preprocessor: Optional[nn.Module] = None,
        spec_augment: Optional[nn.Module] = None,
    ):
        super().__init__(config=TransformerConfig(num_layers=1, hidden_size=1, num_attention_heads=16))
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.spec_augmentation = spec_augment

    def maybe_preprocess_audio(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self.__class__} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        return processed_signal, processed_signal_length

    def forward(
        self,
        input_signal: Optional[torch.Tensor] = None,
        input_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(padded_input=processed_signal, input_lengths=processed_signal_length)

        return encoded, encoded_len


@dataclass
class ASRModuleConfig(ModelParallelConfig, io.IOMixin):
    _target_: Optional[str] = None
    config: Optional[dict] = None
    preprocessor_config: Optional[dict] = None
    spec_augment_config: Optional[dict] = None
    init_from_ptl_ckpt: Optional[str] = None
    sample_rate: Optional[int] = 16000

    def configure_asr_model(self):
        imported_cls = model_utils.import_class_by_path(self._target_)

        cfg = OmegaConf.create(self.config)
        model = imported_cls(**cfg)  # type: nemo_asr.models.ASRModel
        if self.init_from_ptl_ckpt:
            state_dict = torch.load(self.init_from_ptl_ckpt,weights_only=False)["model_state_dict"]
            state_dict = {k.replace('encoder.',''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            model.load_state_dict(state_dict, strict=False)

        preprocessor = Serialization.from_config_dict(to_dict_config(self.preprocessor_config))

        if self.sample_rate != preprocessor._sample_rate:
            raise ValueError(
                f"Sample rate mismatch: ASRModuleConfig ({self.sample_rate}) != preprocessor ({preprocessor._sample_rate}). "
                "Please provide a preprocessor config with the correct sample rate."
            )
        return model, preprocessor


    def configure_model(self):

        model, preprocessor = self.configure_asr_model()

        if self.spec_augment_config is not None:
            spec_augment = Serialization.from_config_dict(to_dict_config(self.spec_augment_config))
        else:
            spec_augment = None

        return MCoreASRModule(encoder=model, preprocessor=preprocessor, spec_augment=spec_augment)
