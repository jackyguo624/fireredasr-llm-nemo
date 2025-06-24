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

import inspect
from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.asr.modules import ConformerEncoder
from nemo.core.classes.module import NeuralModule
from nemo.lightning import io
from nemo.utils import model_utils
from omegaconf import OmegaConf


class MCoreModalityAdapterModule(MegatronModule):
    """
    Wrapper class for modality adapter such as `nemo.collections.asr.modules.ConformerEncoder`.

    `TransformerConfig` is a dummy config to satisfy the `MegatronModule` constructor.
    `num_attention_heads` is set to 16 such that it's divisible by the value of TP.
    `num_layers` and `hidden_size` are set to 1 since not used.
    """

    def __init__(
        self,
        module: NeuralModule,
    ):
        super().__init__(config=TransformerConfig(num_layers=1, hidden_size=1, num_attention_heads=16))
        self.module = module

    def forward(self, encoded, encoded_len):
        if len(inspect.signature(self.module.forward).parameters) == 1:
            return self.module(encoded), encoded_len

        encoded_out, encoded_len_out = self.module(encoded, encoded_len)

        if isinstance(self.module, ConformerEncoder):
            encoded_out = encoded_out.transpose(1, 2)  # (b,d,t) -> (b,t,d)
        return encoded_out, encoded_len_out


@dataclass
class ModalityAdapterConfig(ModelParallelConfig, io.IOMixin):
    _target_: Optional[str] = None
    config: Optional[dict] = None
    init_from_ptl_ckpt: Optional[str] = None

    def configure_model(self):
        imported_cls = model_utils.import_class_by_path(self._target_)
        cfg = OmegaConf.create(self.config)
        module = imported_cls(**cfg)
        if self.init_from_ptl_ckpt:
            state_dict = torch.load(self.init_from_ptl_ckpt, weights_only=False)["model_state_dict"]
            state_dict = {k.replace('encoder_projector.',''): v for k, v in state_dict.items() if k.startswith('encoder_projector.')}
            module.load_state_dict(state_dict)
        return MCoreModalityAdapterModule(module)
