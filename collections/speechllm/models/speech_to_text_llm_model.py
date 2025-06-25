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


import json
import os
import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_only
from megatron.core import dist_checkpointing, parallel_state, tensor_parallel


def safe_tensor_to_numpy(tensor):
    """
    Safely convert a PyTorch tensor to numpy array, handling BFloat16 type.
    BFloat16 is not supported by numpy, so we convert it to float32 first.
    """
    if tensor.dtype == torch.bfloat16:
        return tensor.float().cpu().numpy()
    else:
        return tensor.cpu().numpy()
from megatron.core.inference_params import InferenceParams
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_num_microbatches,
    reconfigure_num_microbatches_calculator,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.eval_utils import remove_punctuations
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.metrics import MetricStringToTorchMetric, TextMetricsSet
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn
from nemo.collections.llm.gpt.model.base import (
    GPTConfig,
    get_batch_on_this_context_parallel_rank,
    get_packed_seq_params,
)
from nemo.collections.speechlm.data.dataset.data_utils import build_position_ids, pad_or_trim_to_max_length
from nemo.collections.speechlm.models.base import SpeechLanguageModel
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig, HFWrappedEncoder
from nemo.collections.speechlm.modules.modality_adapter import ModalityAdapterConfig
from nemo.collections.speechlm.utils.io import get_nested_attr, import_ckpt, load_distributed_ckpt
from fireredasr_llm.collections.speechllm.utils.text_generation.audio_text_generation_strategy import (
    SpeechToTextGenerationStrategy,
)
from fireredasr_llm.collections.speechllm.utils.text_generation.audio_text_generation_utils import (
    clean_end_string,
    default_inference_config,
    generate,
    get_computeprob_response,
)

from nemo.lightning import io
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import AppState, logging, model_utils
from nemo.utils.get_rank import get_last_rank
from transformers import AutoTokenizer

from transformers.trainer_pt_utils import LabelSmoother
DEFAULT_SPEECH_TOKEN = "<speech>"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def set_input_tensor(self, tensor: torch.Tensor):
    """
    Placeholder function for pipeline parallel, not implemented yet.
    """
    pass

def dump_tensor_to_file(tensor_to_save, name, batch=None):
    """
    Dump tensor to text files with 6 decimal places.
    Each sample in the batch is saved to a separate file.
    The first line contains the shape information.

    Args:
        tensor_to_save: Tensor to save (batch_size, ...)
        name: String indicating which feature this is
        batch: Optional batch dict containing keys for naming files
    """
    import numpy as np

    # Create dump directory if it doesn't exist
    dump_dir = "/hpc_stor01/home/jiaqi.guo/tools/github/NeMo/outputs_dump2"
    os.makedirs(dump_dir, exist_ok=True)

    # Get batch keys if available
    if batch is not None and "keys" in batch:
        keys = batch["keys"]
    else:
        keys = [f"sample_{i}" for i in range(tensor_to_save.size(0))]

    # Dump each sample separately
    for i, key in enumerate(keys):
        # Extract single sample data
        sample_tensor = tensor_to_save[i]

        # Create filename
        dump_file = os.path.join(dump_dir, f"{key}_{name}.txt")

        # Convert to numpy and get original shape
        tensor_np = safe_tensor_to_numpy(sample_tensor)

        original_shape = tensor_np.shape

        # Reshape to 2D if needed (flatten all dimensions except the last one)
        if tensor_np.ndim > 2:
            tensor_np = tensor_np.reshape(-1, tensor_np.shape[-1])
        elif tensor_np.ndim == 1:
            tensor_np = tensor_np.reshape(-1, 1)

        # Save with shape on first line
        with open(dump_file, 'w') as f:
            # Write shape as first line
            f.write(f"# Shape: {original_shape}\n")
            # Save tensor data with 6 decimal places
            np.savetxt(f, tensor_np, fmt='%.6f')

        print(f"Dumped {name} for {key} to {dump_file}")

def speech_to_text_llm_data_step(dataloader_iter) -> Dict[str, Any]:
    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842
    # used in SpeechToTextLLMConfig
    batch = next(dataloader_iter)
    _batch: dict
    batch_idx, dataloader_idx = None, None

    if isinstance(batch, tuple) and len(batch) == 3:
        _batch, batch_idx, dataloader_idx = batch
    else:
        _batch = batch

    required_keys = set(
        [
            "sample_ids",
            "position_ids",
            "metadata",
            "inference_params",
            "max_length",
            "contexts",
            "context_lengths",
        ]
    )
    # "context", "context_length", "answers", "max_length",
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(
            (
                "audio_signal",
                "audio_signal_length",
                "processed_signal",
                "processed_signal_length",
                "tokens",
                "tokens_length",
                "context_start_idx",
                "num_audios",
                "answers",
                "contexts",
                "context_lengths",
            )
        )
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))

    _batch = {
        key: move_data_to_device(val, "cuda", non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }

    # inject num_valid_tokens_in_ub for context parallelism,
    # which refers to the total number of valid tokens in the current batch
    if parallel_state.get_context_parallel_world_size() > 1:
        num_valid_tokens_in_ub = None
        if "loss_mask" in _batch and _batch["loss_mask"] is not None:
            num_valid_tokens_in_ub = _batch["loss_mask"].sum()
        _batch["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub

    _batch["dataloader_idx"] = dataloader_idx
    _batch["batch_idx"] = batch_idx
    return _batch


def speech_to_text_llm_forward_step(model: pl.LightningModule, batch: Dict[str, Any]) -> torch.Tensor:
    forward_args = {
        "input_ids": batch.get("tokens", None),
        "input_length": batch.get("tokens_length", None),
        "audio_signal": batch.get("audio_signal", None),
        "audio_signal_length": batch.get("audio_signal_length", None),
        "processed_signal": batch.get("processed_signal", None),
        "processed_signal_length": batch.get("processed_signal_length", None),
        # "context_ids": batch.get("context_ids", None),
        # "context_lengths": batch.get("context_lengths", None),
        "labels": batch.get("labels", None),
        "inference_params": batch.get("inference_params", None),
    }

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)




@dataclass
class SpeechToTextLLMConfig(TransformerConfig, io.IOMixin):

    num_layers: int = 1  # added to avoid init error, not used!!!
    hidden_size: int = 1  # added to avoid init error, not used!!!
    num_attention_heads: int = 16  # added to avoid init error, not used!!!
    seq_length: int = 1024  # added to avoid init error, not used!!!

    language_model_config: Optional[GPTConfig] = None
    language_model_class: Optional[str] = None
    speech_model_config: Optional[ASRModuleConfig] = None
    modality_adapter_config: Optional[ModalityAdapterConfig] = None

    language_model_from_pretrained: Optional[str] = None
    language_model_hub: Optional[str] = 'hf://'

    freeze_language_model: bool = True
    freeze_speech_model: bool = True
    freeze_modality_adapter: bool = False

    forward_step_fn: Callable = speech_to_text_llm_forward_step
    data_step_fn: Callable = speech_to_text_llm_data_step

    text_generation_strategy: SpeechToTextGenerationStrategy = SpeechToTextGenerationStrategy

    inference_config: Optional[Dict[str, Any]] = None

    data_config: Optional[DictConfig] = None

    resume_speech_model_from_path: Optional[str] = None
    resume_modality_adapter_from_path: Optional[str] = None

    def _freeze_module(self, module: Optional[nn.Module] = None) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = False

    def _maybe_load_pretrained_llm(self, model: MCoreGPTModel) -> MCoreGPTModel:
        if not self.language_model_from_pretrained:
            return model

        logging.info(f"Loading language model weights from {self.language_model_from_pretrained}")

        ckpt_path = self.language_model_from_pretrained

        if dist_checkpointing.check_is_distributed_checkpoint(ckpt_path):
            return ckpt_path

        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed environment is not initialized.")

        rank = torch.distributed.get_rank()
        # Sleep to avoid racing condition when multiple GPUs try to import the same checkpoint
        time.sleep(rank / 2)

        llm_model_cls = model_utils.import_class_by_path(self.language_model_class)  # type: GPTModel
        ckpt_path = import_ckpt(
            llm_model_cls(self.language_model_config), f"{self.language_model_hub}{ckpt_path}", on_import_ckpt=False
        )

        sharded_state_dict = dict(state_dict=model.sharded_state_dict(prefix="module."))

        loaded_state_dict = dist_checkpointing.load(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir=ckpt_to_weights_subdir(ckpt_path, is_saving=False),
            validate_access_integrity=False,
        )
        loaded_state_dict = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}
        model.load_state_dict(loaded_state_dict)
        logging.info(f"Restored language model weights from {self.language_model_from_pretrained}")
        return model

    def _maybe_load_asr_and_modality_adapter(
        self, asr_model: ASRModel, modality_adapter: nn.Module
    ) -> Tuple[ASRModel, nn.Module]:
        if self.resume_speech_model_from_path:
            logging.info(f"Loading speech model weights from {self.resume_speech_model_from_path}")
            state_dict, _ = load_distributed_ckpt(self.resume_speech_model_from_path)
            prefix = 'module.speech_model.'
            speech_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            asr_model.load_state_dict(speech_state_dict, strict=True)
            logging.info(f"Restored speech model weights from {self.resume_speech_model_from_path}")

        if self.resume_modality_adapter_from_path:
            logging.info(f"Loading modality adapter weights from {self.resume_modality_adapter_from_path}")
            state_dict, _ = load_distributed_ckpt(self.resume_modality_adapter_from_path)
            prefix = 'module.modality_adapter.'
            modality_adapter_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            modality_adapter.load_state_dict(modality_adapter_state_dict, strict=True)
            logging.info(f"Restored modality adapter weights from {self.resume_modality_adapter_from_path}")

        return asr_model, modality_adapter

    def _propagate_model_configs(self) -> TransformerConfig:
        """
        propagate key attributes to the language/speech model config
        """
        # LLM
        self.language_model_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.language_model_config.sequence_parallel = self.sequence_parallel
        self.language_model_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.language_model_config.context_parallel_size = self.context_parallel_size

        # ASR
        self.speech_model_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.speech_model_config.sequence_parallel = self.sequence_parallel
        self.speech_model_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.speech_model_config.context_parallel_size = self.context_parallel_size

        # modality adapter
        self.modality_adapter_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.modality_adapter_config.sequence_parallel = self.sequence_parallel
        self.modality_adapter_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
        self.modality_adapter_config.context_parallel_size = self.context_parallel_size

    def configure_model(
        self, tokenizer: TokenizerSpec, speech_model: Optional[ASRModel] = None
    ) -> "MCoreSpeechToTextLLM":
        self._propagate_model_configs()
        language_model = self.language_model_config.configure_model(tokenizer=tokenizer)  # type: "MCoreGPTModel"
        language_model = self._maybe_load_pretrained_llm(language_model)

        if speech_model is None:
            # propagate key attributes to the speech model config
            speech_model = self.speech_model_config.configure_model()  # type: MCoreASRModel
        speech_model.set_input_tensor = MethodType(set_input_tensor, speech_model)

        self.modality_adapter_config.llm_dim = self.language_model_config.hidden_size

        modality_adapter = self.modality_adapter_config.configure_model()
        modality_adapter.set_input_tensor = MethodType(set_input_tensor, modality_adapter)

        speech_model, modality_adapter = self._maybe_load_asr_and_modality_adapter(speech_model, modality_adapter)
        model = MCoreSpeechToTextLLM(
            config=self,
            language_model=language_model,
            speech_model=speech_model,
            modality_adapter=modality_adapter,
            tokenizer=tokenizer,
        )

        if self.freeze_language_model:
            self._freeze_module(model.language_model)
        if self.freeze_speech_model:
            self._freeze_module(model.speech_model)
        if self.freeze_modality_adapter:
            self._freeze_module(model.modality_adapter)

        return model


class MCoreSpeechToTextLLM(MegatronModule, fn.FNMixin):
    def __init__(
        self,
        config: SpeechToTextLLMConfig,
        language_model: MegatronModule,
        speech_model: ASRModel,
        modality_adapter: nn.Module,
        tokenizer: AutoTokenizer,
    ):
        super().__init__(config=config)
        self.language_model = language_model
        self.speech_model = speech_model
        self.modality_adapter = modality_adapter
        self.tokenizer = tokenizer
        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        self._language_max_sequence_length = self.language_model.max_sequence_length

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        NOTE: Pipeline parallelism is not supported in this model yet. This is just a placeholder implementation.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        pass

    def _create_attention_mask(self, encoder_input: torch.Tensor):
        """
        Create causal attention mask for whole input
        Args:
            encoder_input: The encoder input tensor of shape [b, t, h].
        Returns:
            attention_mask: The attention mask tensor of shape [b, 1, t, t].
        """
        # Create causal attention mask for whole input
        batch_size = encoder_input.shape[0]
        max_len = encoder_input.shape[1]
        attention_mask = torch.tril(torch.ones((batch_size, max_len, max_len), device=encoder_input.device)).view(
            batch_size, 1, max_len, max_len
        )
        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5
        # [batch, 1, seq_len, seq_len]
        return attention_mask

    def inject_perception_input(
        self,
        speech_features: torch.Tensor,
        speech_lens: torch.Tensor,
        input_ids: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        # [b, t, c]
        lm_embedding = self.language_model.embedding
        inputs_embeds = lm_embedding.word_embeddings(input_ids)

        default_speech_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        num_speechs, speech_len, embed_dim = speech_features.shape

        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(pad_token_id)
        )
        # 1. Create a mask to know where special speech tokens are
        special_speech_token_mask = input_ids == default_speech_token_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_speech_tokens.max() * (speech_len - 1)
        ) + sequence_length
        batch_indices, non_speech_indices = torch.where(
            input_ids != default_speech_token_id
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged speech-text sequence.
        # `special_speech_token_mask` identifies speech tokens. Each speech token will be replaced by `nb_text_tokens_per_speechs - 1` text tokens.
        # `torch.cumsum` computes how each speech token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1
        )  # (N,U)
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions = new_token_positions + nb_speech_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]


        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                IGNORE_TOKEN_ID,
                dtype=input_ids.dtype,
                device=input_ids.device,
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_speech_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_speech_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<speech>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the speech features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_speech_indices
        ]

        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_speech_indices
            ]
            # Create loss mask - initially all False
            final_loss_mask = torch.zeros(
                (batch_size, max_embed_dim),
                dtype=torch.bool,
                device=input_ids.device,
            )
            # Set loss mask to True only for positions with actual labels (not IGNORE_TOKEN_ID)
            # and not for speech feature positions
            valid_label_mask = labels[batch_indices, non_speech_indices] != IGNORE_TOKEN_ID
            final_loss_mask[batch_indices[valid_label_mask], text_to_overwrite[valid_label_mask]] = True


        # 5. Fill the embeddings corresponding to the speechs. Anything that is not `text_positions` needs filling (#29835)
        speech_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False

        speech_pad_position = speech_to_overwrite.cumsum(-1) <= speech_lens[:, None]

        speech_to_overwrite = speech_to_overwrite & (speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[
            :, None
        ].to(target_device))

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while"
                f" the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[speech_to_overwrite] = (
            speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )

        speech_to_overwrite = speech_to_overwrite & speech_pad_position

        final_embedding_length = input_length + speech_lens

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(
            input_ids == pad_token_id
        )
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        # If the input flag for fp32 residual connection is set, convert for float.
        if lm_embedding.config.fp32_residual_connection:
            combined_embed = combined_embed.float()

        if labels is None:
            final_labels = None
            final_loss_mask = None

        position_ids = build_position_ids(final_embedding[:, :, 0])

        return final_embedding, final_labels, final_loss_mask, final_embedding_length, position_ids

    def _get_text_embeddings(self, text_tokens, position_ids):
        """Get text embeddings for the input text tokens for inference decoding."""
        lm_embedding = self.language_model.embedding
        text_embeddings = lm_embedding.word_embeddings(text_tokens)  # (batch_size, seq_len, hidden_size)
        if hasattr(lm_embedding, 'position_embeddings'):
            position_embeddings = lm_embedding.position_embeddings(position_ids)
            text_embeddings = text_embeddings + position_embeddings

        text_embeddings = text_embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if lm_embedding.config.fp32_residual_connection:
            text_embeddings = text_embeddings.float()

        # Dropout.
        if lm_embedding.config.sequence_parallel:
            text_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(text_embeddings)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if lm_embedding.config.clone_scatter_output_in_embedding:
                text_embeddings = text_embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                text_embeddings = lm_embedding.embedding_dropout(text_embeddings)
        else:
            text_embeddings = lm_embedding.embedding_dropout(text_embeddings)

        return text_embeddings

    def _get_llm_input_for_context_parallel(
        self,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor,
        labels: torch.Tensor,
        loss_masks: torch.Tensor,
        max_length: int,
    ):
        """
        Prepare context parallel input for the language model, where tensors are padded to the lengths
        divisible by context parallel world size.
        Args:
            attention_mask: The attention mask tensor of shape [b, 1, t, t].
            decoder_input: The decoder input tensor of shape [t, b, h].
            labels: The labels tensor of shape [b, t].
            loss_masks: The loss mask tensor of shape [b, t].
            max_length: The maximum length of the input tensors, integer.
        Returns:
            attention_mask_cp: The attention mask tensor for context parallelism, shape [b, 1, t, t].
            decoder_input_cp: The decoder input tensor for context parallelism, shape [t, b, h].
            labels_cp: The labels tensor for context parallelism, shape [b, t].
        """
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            return attention_mask, decoder_input, labels, loss_masks

        shard_factor = 2 * cp_size  # 2x required by megatron context parallel
        decoder_input = decoder_input.transpose(0, 1).contiguous()  # [t, b, h] -> [b, t, h]
        decoder_input = pad_or_trim_to_max_length(decoder_input, max_length, 0, ceil_to=shard_factor)
        labels = pad_or_trim_to_max_length(labels, max_length, 0, ceil_to=shard_factor)
        loss_masks = pad_or_trim_to_max_length(loss_masks, max_length, 0, ceil_to=shard_factor)
        attention_mask = self._create_attention_mask(decoder_input)

        batch = {
            "attention_mask": attention_mask,
            "decoder_input": decoder_input,
            "labels": labels,
            "loss_mask": loss_masks,
        }

        # Split the batch for context parallelism
        batch_cp = get_batch_on_this_context_parallel_rank(batch)
        attention_mask_cp = batch_cp["attention_mask"]
        decoder_input_cp = batch_cp["decoder_input"].transpose(0, 1).contiguous()  # [b, t, h] -> [t, b, h]
        labels_cp = batch_cp["labels"]
        loss_masks_cp = batch_cp["loss_mask"]
        return attention_mask_cp, decoder_input_cp, labels_cp, loss_masks_cp

    def perception(self, input_signal, input_signal_length, processed_signal, processed_signal_length):
        encoded, encoded_len = self.speech_model(
            input_signal, input_signal_length,
        )
        encoded, encoded_len = self.modality_adapter(encoded, encoded_len)

        return encoded, encoded_len

    def forward(
        self,
        input_ids: torch.Tensor,
        input_length: torch.Tensor,
        audio_signal: Optional[torch.Tensor] = None,
        audio_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[Dict[str, Any]] = None,
    ):

        encoded, encoded_len = self.perception(
            input_signal=audio_signal,
            input_signal_length=audio_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

        (
            final_embedding,
            final_labels,
            final_loss_mask,
            final_embedding_length,
            position_ids
        ) = self.inject_perception_input(
            speech_features=encoded,
            speech_lens=encoded_len,
            input_ids=input_ids,
            input_length=input_length,
            labels=labels,
        )


        final_attention_mask = self._create_attention_mask(final_embedding)
        final_embedding = final_embedding.transpose(0, 1).contiguous()

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=final_attention_mask.bool(),
            decoder_input=final_embedding,
            labels=final_labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
        )

        if final_labels is None or final_loss_mask is None:
            return output

        # Return the output (language model already handles loss calculation internally)
        return output, final_loss_mask.contiguous()


class SpeechToTextLLM(SpeechLanguageModel):
    def __init__(
        self,
        config: SpeechToTextLLMConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None
        self._inference_config = None
        self._speech_model = self.config.speech_model_config.configure_model()

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer, self._speech_model)  # type: MCoreSpeechToTextLLM
            self.module.language_model = self.module.language_model.to(self.device)
            self.module.speech_model = self.module.speech_model.to(self.device)
            self.module.modality_adapter = self.module.modality_adapter.to(self.device)
            del self._speech_model
            torch.cuda.empty_cache()


    def setup(self, stage: str):
        super().setup(stage)
        if hasattr(self.cfg.data, "validation_ds"):
            self.val_metric, self.val_metric_name = self.setup_metric(self.cfg.data.validation_ds)
            self.val_metric = torch.nn.ModuleList(self.val_metric) if self.val_metric is not None else None
            # Used other keys from metadata to calulate metrics
            if hasattr(self.cfg.data.validation_ds, "metric"):
                self.val_metric_label_key = self.cfg.data.validation_ds.metric.get('label_key', 'labels')

        if hasattr(self.cfg.data, "test_ds"):
            self.test_metric, self.test_metric_name = self.setup_metric(self.cfg.data.test_ds)
            self.test_metric = torch.nn.ModuleList(self.test_metric) if self.test_metric is not None else None
            # Used other keys from metadata to calulate metrics
            if hasattr(self.cfg.data.test_ds, "metric"):
                self.test_metric_label_key = self.cfg.data.test_ds.metric.get('label_key', 'labels')

        if self.get_inference_config() is None:
            self.set_inference_config(self.config.inference_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_length: torch.Tensor,
        audio_signal: Optional[torch.Tensor] = None,
        audio_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output = self.module(
            input_ids=input_ids,
            input_length=input_length,
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            labels=labels,
            inference_params=inference_params,
        )
        return output

    def freeze_llm(self):
        module = self.module
        while not hasattr(module, "language_model"):
            module = module.module
        self.freeze_module(module.language_model)

    def freeze_speech(self):
        module = self.module
        while not hasattr(module, "speech_model"):
            module = module.module
        self.freeze_module(module.speech_model)

    def freeze_modality_adapter(self):
        module = self.module
        while not hasattr(module, "modality_adapter"):
            module = module.module
        self.freeze_module(module.modality_adapter)

    def unfreeze_llm(self):
        module = self.module
        while not hasattr(module, "language_model"):
            module = module.module
        self.unfreeze_module(module.language_model)

    def unfreeze_speech(self):
        module = self.module
        while not hasattr(module, "speech_model"):
            module = module.module
        self.unfreeze_module(module.speech_model)

    def unfreeze_modality_adapter(self):
        module = self.module
        while not hasattr(module, "modality_adapter"):
            module = module.module
        self.unfreeze_module(module.modality_adapter)

    def trainable_parameters(self) -> List[Tuple[str, torch.Tensor]]:
        """
        This function returns all trainable parameters of the model,
        including some params that don't require gradients (e.g., batchnorm).
        This function is used for PEFT to determine what parameters to load/save.
        See `nemo/collections/speechlm/utils/model_transform.py` for more details.
        The name of this function is set to align with the PEFT API.
        """
        trainable_params = []
        # must use state_dict() to include params like batchnorm running mean/var
        for name, param in self.state_dict().items():
            if name.startswith("module.speech_model.") and not self.config.freeze_speech_model:
                trainable_params.append((name, param))
            elif name.startswith("module.modality_adapter.") and not self.config.freeze_modality_adapter:
                trainable_params.append((name, param))
            elif name.startswith("module.language_model.") and not self.config.freeze_language_model:
                trainable_params.append((name, param))
            elif (
                name.startswith("module.language_model.")
                and self.config.freeze_language_model
                and (".adapter." in name or name.endswith(".adapters"))
            ):
                trainable_params.append((name, param))

        return trainable_params

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        return self.inference_step(batch, mode='validation')

    @property
    def _metrics_require_string2category_map(self):
        return set(["f1", "accuracy", "average_precision"])

    def setup_metric(self, data_cfg):
        metric_name = "exact_string_match"
        if not hasattr(data_cfg, "metric"):
            metric = MetricStringToTorchMetric["exact_string_match"]
        else:
            if not hasattr(data_cfg.metric, "name"):
                raise ValueError("Metric name is not provided in the metric config.")
            if data_cfg.metric.name == "loss":
                return None, "loss"
            if data_cfg.metric.name not in MetricStringToTorchMetric:
                raise KeyError(
                    f"{data_cfg.metric.name} is not supported. List of supported metrics: {MetricStringToTorchMetric.keys()}"
                )
            if data_cfg.metric.name in self._metrics_require_string2category_map:
                if data_cfg.metric.average is None:
                    raise ValueError(
                        f"{data_cfg.metric.name} requires specifying whether you want to compute a micro or macro average. Found None."
                    )
            if (
                data_cfg.metric.get('labels_are_strings', False)
                and data_cfg.metric.name in self._metrics_require_string2category_map
            ):
                if data_cfg.metric.num_classes is None:
                    raise ValueError(
                        "Number of classes is not provided in the metric section within the data config. "
                        f"Please provide the number of classes in the data config to use the {data_cfg.metric.name} metric."
                    )
                if data_cfg.metric.get('class_labels', None) is None or not isinstance(
                    data_cfg.metric.get('class_labels', None), ListConfig
                ):
                    raise ValueError(
                        "Class labels are not provided properly in the metric section witnin the data config. "
                        f"Please provide the class labels as a list of strings in the data config to use the {data_cfg.metric.name} metric."
                    )
                if len(data_cfg.metric.get('class_labels', None)) != data_cfg.metric.num_classes:
                    raise ValueError(
                        f"Number of class labels {len(data_cfg.metric.get('class_labels', None))} does not match `num_classes` : {data_cfg.metric.num_classes}"
                    )

            metric_name = data_cfg.metric.name
            metric_cls = MetricStringToTorchMetric[metric_name]
            if metric_name not in TextMetricsSet:
                metric = [metric_cls(**data_cfg.metric)]
            else:
                metric = [metric_cls()]
        return metric, metric_name

    def inference_step(self, batch, mode):
        """
        Used for validation and test steps, added postprocessing after calling self.predict_step().
        """
        batch_idx = batch.pop("batch_idx", None)
        dataloader_idx = batch.pop("dataloader_idx", None)

        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds

        self._reconfigure_and_process_inference_batch(batch, data_cfg)
        # Meta data from dataset
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))

        forward_output = self.forward_step(batch)

        if isinstance(forward_output, tuple):
            # reduce validation loss
            loss = self.validation_loss_reduction.forward(batch=batch, forward_out=forward_output)[-1]['avg']
        else:
            # no labels provided, use a dummy loss value
            loss = 0.0

        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        preds_text = []
        labels_text = []
        inputs_text = []
        if metric_name != "loss":
            # We need _inference_config to get generation params, tokens_to_generate are set in dataset
            if self.get_inference_config() is None:
                logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
                self.set_inference_config(inference_config=default_inference_config)
            self._inference_config['tokens_to_generate'] = data_cfg.get('tokens_to_generate')

            output = self.predict_step(batch, batch_idx, dataloader_idx)
            from fireredasr_llm.collections.speechllm.utils.text_generation.audio_text_generation_utils import safe_decode
            # inputs_text = [safe_decode(self.tokenizer, c) for c in batch['contexts']]
            inputs_text = [safe_decode(self.tokenizer, c[:l.item()], skip_special_tokens=True)
                           for c, l in zip(batch['contexts'], batch['context_lengths'])]
            labels_text = [a for a in batch['answers']]
            preds_text = [safe_decode(self.tokenizer, t[l.item():][: data_cfg.get('tokens_to_generate')], skip_special_tokens=True)
                for t, l in zip(output['token_ids'], batch['context_lengths'])]

            if data_cfg.get("end_string", None):
                # sometimes data_cfg.end_string != self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(data_cfg.end_string))
                # for example when data_cfg.end_string = "<end>", the end_string_re will start with " ?? "
                preds_text = clean_end_string(preds_text, self.tokenizer, data_cfg.end_string)
                labels_text = clean_end_string(labels_text, self.tokenizer, data_cfg.end_string)

            if data_cfg.get("remove_text_pc", False):
                preds_text = [
                    remove_punctuations(pred_text.lower(), data_cfg.get("punctuations", None))
                    for pred_text in preds_text
                ]
                labels_text = [
                    remove_punctuations(label_text.lower(), data_cfg.get("punctuations", None))
                    for label_text in labels_text
                ]

            if data_cfg.get("log_every_n_steps", None) is not None:
                if batch_idx % data_cfg.log_every_n_steps == 0:
                    logging.info(f"Input: `{inputs_text[0]}`")
                    logging.info(f"Label: `{labels_text[0]}`")
                    logging.info(f"Pred: `{preds_text[0]}`")

        outputs = {
            'loss': loss,
            'preds': preds_text,  # [str]
            'labels': labels_text,  # [str]
            'inputs': inputs_text,  # [str]
            'metadata': metadata,  # [dict]
        }

        if mode == 'validation':
            if self._num_validation_dl > 1:
                self.validation_step_outputs[dataloader_idx].append(outputs)
            else:
                self.validation_step_outputs.append(outputs)
        else:
            if self._num_test_dl > 1:
                self.test_step_outputs[dataloader_idx].append(outputs)
            else:
                self.test_step_outputs.append(outputs)
        return forward_output

    def get_inference_strategy(self):
        return self.config.text_generation_strategy(self.module)

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None):
        """
        Used to get LLM predictions for validation and test steps based on the given inference config.
        """
        inference_config = self.get_inference_config()
        if inference_config is not None:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
        else:
            self.set_inference_config(inference_config=default_inference_config)
            logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
            inference_config = self.get_inference_config()

        if self.cfg.data.get('end_string', None):
            inference_config['end_strings'] = [self.cfg.data.end_string]

        inference_config['strategy'] = self.get_inference_strategy()

        global_batch_size_per_gpu = batch['tokens'].size(0)
        num_micro_batches_before_decode = get_num_microbatches()

        compute_logprob = inference_config.get('compute_logprob', False)

        if compute_logprob:
            inference_config['inputs'] = batch
            inference_config['tokens_to_generate'] = 1
            inference_config['all_probs'] = True
            inference_config['greedy'] = True
            response = generate(self, **inference_config)
            response = get_computeprob_response(self.tokenizer, response, batch)
        else:
            if isinstance(batch, list):
                inference_config['inputs'] = batch
            else:
                inference_config['inputs'] = (
                    batch['contexts'].cuda(),
                    batch['context_lengths'].cuda(),
                    batch['audio_signal'].cuda(),
                    batch['audio_signal_length'].cuda(),
                )
            response = generate(self, **inference_config)

        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_size_per_gpu // num_micro_batches_before_decode,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        # add audio offsets to context lengths for properly decoding only the response
        batch['context_lengths'] = batch['context_lengths'].cuda() + response['audio_feat_lens']

        return response

    def _determine_log_key(self, dataloader_idx, metric_name, mode):
        # If the user provided names for each validation/test dataset, use those.
        if mode == 'validation':
            prefix = self.get_validation_dataloader_prefix(dataloader_idx)
            if prefix.startswith('val_'):
                # no user provided name, use the dataloader idx
                log_key = f'val_{metric_name}_{dataloader_idx}'
            else:
                log_key = f'val_{metric_name}_{prefix}'
        else:
            prefix = self.get_test_dataloader_prefix(dataloader_idx).strip('test_')
            if prefix.startswith('test_'):
                # no user provided name, use the dataloader idx
                log_key = f'test_{metric_name}_{dataloader_idx}'
            else:
                log_key = f'test_{metric_name}_{prefix}'
        return log_key

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # Parent class will handle logging of the loss.
        if not outputs or (all([not x for x in outputs])):
            return None

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_metric = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            if len(output) == 0:
                logging.warning(f"Empty output for dataloader_idx: {dataloader_idx}")
                continue
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            loss_vals = [x['loss'].view(-1, 1) for x in output]  # each loss is [1, B]
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                loss = torch.vstack(loss_vals).mean().type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(dataloader_idx, "loss", mode)
            self.log(loss_log_key, loss, batch_size=1)
            averaged_loss.append(loss)

            metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
            if metric_name != 'loss':
                self.gather_and_maybe_write_predictions(data_cfg, output, averaged_metric, mode, dataloader_idx)

            torch.distributed.barrier(group=parallel_state.get_data_parallel_group())
            outputs[dataloader_idx].clear()  # free memory

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric) if len(averaged_metric) > 0 else None
        averaged_loss = averaged_loss.to(self.device)
        if averaged_metric is not None:
            averaged_metric = averaged_metric.to(self.device)

        # Handle case where metrics can be nan or inf. This can break checkpoint save/load.
        if averaged_metric is not None and (torch.isinf(averaged_metric) or torch.isnan(averaged_metric)):
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e5

        if mode == 'validation':
            self.log("val_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"val_{self.val_metric_name}", averaged_metric, sync_dist=True, batch_size=1)
        elif mode == 'test':
            self.log("test_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"test_{self.test_metric_name}", averaged_metric, sync_dist=True, batch_size=1)

        # Merge the functionality of previous on_inference_epoch_end() within inference_epoch_end() func here
        app_state = AppState()
        # self._restore_activation_checkpointing_args()
        if hasattr(self.cfg.data, "train_ds"):
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=data_cfg.global_batch_size,
                micro_batch_size=data_cfg.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        return averaged_loss, averaged_metric

    def gather_and_maybe_write_predictions(self, data_cfg, output, averaged_metric, mode, dataloader_idx):
        # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
        gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            gathered_outputs,
            [
                {'preds': x['preds'], 'labels': x['labels'], 'inputs': x['inputs'], 'metadata': x['metadata']}
                for x in output
            ],
            group=parallel_state.get_data_parallel_group(),
        )

        # Remove duplicate examples due to distributed sampler.
        inp_label_set = set()
        deduplicated_outputs = {
            'preds': [],
            'labels': [],
            'inputs': [],
            'metadata': [],
        }
        total_size = 0
        for rank in range(0, parallel_state.get_data_parallel_world_size()):
            for batch in gathered_outputs[rank]:
                for pred, label, input, metadata in zip(
                    batch['preds'], batch['labels'], batch['inputs'], batch['metadata']
                ):
                    key = input + label + str(metadata)
                    total_size += 1
                    if key not in inp_label_set:
                        inp_label_set.add(key)
                        deduplicated_outputs['preds'].append(pred)
                        deduplicated_outputs['labels'].append(label)
                        deduplicated_outputs['inputs'].append(input)
                        deduplicated_outputs['metadata'].append(metadata)

        # Compute metric score
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        metric_label_key = self.val_metric_label_key if mode == 'validation' else self.test_metric_label_key
        if metric_name != 'loss':
            metric_log_key = self._determine_log_key(dataloader_idx, metric_name, mode)
            metric_fn = self.val_metric[0] if mode == 'validation' else self.test_metric[0]
            if metric_label_key in deduplicated_outputs['metadata'][0]:
                labels = [m[metric_label_key] for m in deduplicated_outputs['metadata']]
            else:
                labels = deduplicated_outputs['labels']

            # Compute metrics
            # SacreBLEU does not share the same interface as other metrics. We handle it separately.
            for pred, label in zip(deduplicated_outputs['preds'], labels):
                if metric_name == 'bleu':
                    _ = metric_fn([pred], [[label]])
                else:
                    _ = metric_fn(pred, label)

            metric_result = metric_fn.compute()

            # log the metrics
            if metric_name == 'rouge':
                for k, v in metric_result.items():
                    if 'fmeasure' in k:
                        self.log(metric_log_key + f'_{k}', v.item(), sync_dist=True, batch_size=1)
                        logging.info(f"{metric_log_key}_{k}]: {v.item()}")
                metric_result = metric_result['rouge1_fmeasure']
            else:
                self.log(metric_log_key, metric_result.item(), sync_dist=True, batch_size=1)
                logging.info(f"{metric_log_key}: {metric_result.item()}")

            metric_fn.reset()
            averaged_metric.append(metric_result)

        # Write predictions to file
        if self.global_rank == 0 and data_cfg.get("write_predictions_to_file", False):
            logging.info(
                f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['inputs'])}"
            )

            # Check if the user provided a prefix path to the file(s) they want to write.
            filename_log_key = self._determine_log_key(dataloader_idx, metric_name, mode)
            output_dir = data_cfg.get("output_dir", "./")
            self.write_predictions_to_file(deduplicated_outputs, f"speechlm_pred_{filename_log_key}", output_dir)

    # consistent with speech models
    @rank_zero_only
    def write_predictions_to_file(self, outputs, output_file_path_prefix, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = output_file_path_prefix + "_inputs_preds_labels.jsonl"
        output_file_path = os.path.join(output_dir, output_file_path)
        with open(output_file_path, "w", encoding="utf-8") as f_json:
            assert (
                len(outputs['inputs']) == len(outputs['preds']) == len(outputs['labels']) == len(outputs['metadata'])
            )
            for i, p, l, m in zip(outputs['inputs'], outputs['preds'], outputs['labels'], outputs['metadata']):
                json_string = {'input': i, 'pred_text': p, 'text': l}
                for k, v in m.items():
                    if k not in json_string:
                        json_string[k] = v
                f_json.write(json.dumps(json_string, ensure_ascii=False) + '\n')

        logging.info(f'Predictions saved to {output_file_path}')

    # Override the parent batch reconfiguring logic.
    def _reconfigure_and_process_inference_batch(self, batch, data_cfg):
        global_batch_size_per_gpu = batch['tokens'].size(0)
        # This should happen only on the last batch of the dataset.
        if (
            global_batch_size_per_gpu
            != get_current_global_batch_size() // parallel_state.get_data_parallel_world_size()
        ):
            # NOTE: This is reconfiguring to make sure there is no grad-acc for validation batches.
            if (
                global_batch_size_per_gpu
                != data_cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
            ):
                app_state = AppState()
                reconfigure_num_microbatches_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
                    micro_batch_size=global_batch_size_per_gpu,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            # NOTE: need to explicitly handle resetting for multi-validation
            else:
                app_state = AppState()
                reconfigure_num_microbatches_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=data_cfg.global_batch_size,
                    micro_batch_size=data_cfg.micro_batch_size,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )

    def set_inference_config(self, inference_config: Optional[Dict] = None):
        self._inference_config = dict(inference_config) if inference_config is not None else None

    def get_inference_config(self):
        return dict(self._inference_config) if self._inference_config is not None else None

    def on_validation_epoch_start(self):
        # self._reset_activation_checkpointing_args()
        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.validation_ds.global_batch_size,
            micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        # self._reset_activation_checkpointing_args()
        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.test_ds.global_batch_size,
            micro_batch_size=self.cfg.data.test_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_test_epoch_start()

    def on_predict_epoch_start(self):
        return self.on_test_epoch_start()

    def on_test_epoch_end(self):
        _ = self.inference_epoch_end(self.test_step_outputs, 'test', self.cfg.data.test_ds)
        # Commenting as on_test_epoch_end was a no-op in PTL 1.9
        # return super().on_test_epoch_end()

    def on_validation_epoch_end(self):
        _ = self.inference_epoch_end(self.validation_step_outputs, 'validation', self.cfg.data.validation_ds)
        # Commenting as on_validation_epoch_end was a no-op in PTL 1.9
        # return super().on_validation_epoch_end()

    def on_train_epoch_start(self) -> None:
        # Same logic as validation epoch end, but this may be need if there is no validation sanity check to trigger on_validation_epoch_end()
        self.on_validation_epoch_end()
        return super().on_train_epoch_start()
