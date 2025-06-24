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

from typing import Any, List, Tuple, Union

import torch.utils.data
from lhotse.cut import Cut, CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse
from transformers import AutoTokenizer
from nemo.collections.speechlm.data.dataset.data_utils import build_loss_mask
from nemo.collections.speechlm.data.text_processing import TextProcessorOutput
from fireredasr_llm.collections.speechllm.data.llm_tokenizer import LlmTokenizerWrapper
from lightning.pytorch.utilities.rank_zero import rank_zero_only

def collate_vectors(items, max_length: int, padding_value):
    vectors = collate_vectors_lhotse(items, padding_value=padding_value)
    if max_length > vectors.size(1):
        vectors = torch.cat(
            [vectors, padding_value * torch.ones(vectors.size(0), max_length - vectors.size(1), dtype=vectors.dtype)],
            dim=1,
        )
    if items[0].shape[0] < 1:
        vectors = vectors.long()
    return vectors


class MultimodalConversationDataset(torch.utils.data.Dataset):
    """
    This dataset is based on Lhotse ASR dataset from ``audio_to_text_lhotse.py``
    and ``TarredAudioQuestionAnswerDataset`` from ``audio_text_qa_dataset.py``.

    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.

    Args:
        text_processor: TextProcessing object
        default_context: Default question to use if no question is provided
        tokens_to_generate: Number of tokens to generate during inference
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        max_seq_length: Maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        context_key: Key to use for the context in your JSONL file
        default_context_key: Key to use for the default context in lhotse yaml
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        tokens_to_generate: int,
        max_seq_length: int,
        is_train: bool = False,
    ):
        super().__init__()

        self.load_audio = AudioSamples(fault_tolerant=True)
        self.tokens_to_generate = tokens_to_generate
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

        self.is_train = is_train

    def __getitem__(self, all_cuts: CutSet) -> dict[str, Union[torch.Tensor, list[str], dict]]:

        print(f"len of cuts: {len(all_cuts)}")
        # for cut in all_cuts:
        audio_samples = [
            torch.tensor(cut.load_audio().reshape(-1), dtype=torch.int16)
            for cut in all_cuts
        ]

        audio_length = [len(audio_signal) for audio_signal in audio_samples]

        max_audio_length = max(audio_length)

        text_samples = [
            cut.supervisions[0].text
            for cut in all_cuts
        ]

        tokens, tokens_length, _, target_ids, clean_texts = \
            LlmTokenizerWrapper.preprocess_texts(
                text_samples,
                self.tokenizer,
                max_len=self.max_seq_length,
                decode = False
            )

        context_ids, context_lengths, _, _, _ = \
            LlmTokenizerWrapper.preprocess_texts(
                text_samples,
                self.tokenizer,
                max_len=self.max_seq_length,
                decode = True
            )

        audio_signal_tensor = collate_vectors(audio_samples, max_audio_length, padding_value=0.0)
        audio_length_tensor = torch.tensor(audio_length).long()

        # extend context_ids by tokens_to_generate number of padding tokens
        pad_id = self.tokenizer.pad_token_id
        context_ids = torch.cat(
            [
                context_ids,
                torch.ones(context_ids.size(0),
                          self.tokens_to_generate + 1,
                          dtype=context_ids.dtype,
                          device=context_ids.device) * pad_id
            ],
            dim=1
        )

        sample_ids = list(all_cuts.ids)
        metadata = self._get_metadata(all_cuts)

        batch = {
            "sample_ids": sample_ids,
            "metadata": metadata,
            "audio_signal": audio_signal_tensor,
            "audio_signal_length": audio_length_tensor,
            "tokens": tokens,
            "tokens_length": tokens_length,
            "labels": target_ids,
            "contexts": context_ids,
            "context_lengths": context_lengths,
            "answers": clean_texts,
        }
        return batch

    def _get_metadata(self, all_cuts: CutSet) -> List[dict]:
        metadata = []
        for cut in all_cuts:
            metadata.append({"type": type(cut).__name__, "id": getattr(cut, "id", "n/a")})
        return metadata



