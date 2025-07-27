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

"""Utilities for generating text."""

import pickle
from collections.abc import Iterable
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

import nemo.collections.nlp.modules.common.text_generation_utils as text_generation_utils
from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
from nemo.collections.multimodal.speech_llm.modules.common.audio_text_generation_strategy import (
    model_inference_strategy_dispatcher,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import OutputType
from nemo.utils import AppState, logging

try:
    from megatron.core import parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import reconfigure_num_microbatches_calculator

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator as reconfigure_num_microbatches_calculator,
    )

__all__ = [
    "get_computeprob_response",
    "generate",
    "default_inference_config",
    "clean_end_string",
]


default_inference_config = {'tokens_to_generate': 64}


def clean_end_string(text: list[str], tokenizer, end_string: Optional[str] = None):
    if end_string is None:
        return text

    text_list = [text] if isinstance(text, str) else text
    end_string_re = tokenizer.ids_to_text(tokenizer.text_to_ids(end_string))
    cleaned_text = []
    for t in text_list:
        for es in [end_string_re, end_string]:
            if t.endswith(es):
                t = t[: -len(es)].strip()
        cleaned_text.append(t)
    if isinstance(text, str):
        return cleaned_text[0]
    return cleaned_text


def get_computeprob_response(tokenizer, response, inputs):
    return text_generation_utils.get_computeprob_response(tokenizer, response, inputs)

def synced_generate(
    model,
    inference_strategy,
    context_tokens_tensor,
    context_length_tensor,
    audio_signal,
    audio_signal_length,
    tokens_to_generate,
    all_probs,
    temperature,
    top_k=0,
    top_p=0.0,
    greedy=False,
    compute_attention_mask=True,
    compute_logprob=False,
    repetition_penalty=1.2,
    end_strings=[],
    min_tokens_to_generate=0,
    num_audios: Optional[torch.Tensor] = None,
    context_start_idx: Optional[List[List[int]]] = None,
    beam_size=None,
):
    context_length = context_length_tensor.min().item()
    tokenizer = model.tokenizer
    if isinstance(tokenizer, TabularTokenizer):
        raise NotImplementedError("Tabular generation is not supported yet")
    elif beam_size is None:
        batch_token_iterator = sample_sequence_batch(
            model,
            inference_strategy,
            context_tokens_tensor,
            context_length_tensor,
            audio_signal,
            audio_signal_length,
            tokens_to_generate,
            all_probs,
            compute_attention_mask=compute_attention_mask,
            compute_logprob=compute_logprob,
            temperature=temperature,
            end_strings=end_strings,
            extra={
                "top_p": top_p,
                "top_k": top_k,
                "greedy": greedy,
                "repetition_penalty": repetition_penalty,
                "min_tokens_to_generate": min_tokens_to_generate,
            },
            num_audios=num_audios,
            context_start_idx=context_start_idx,
        )
    else:
        batch_token_iterator = sample_sequence_batch_beamsearch(
            model,
            inference_strategy,
            context_tokens_tensor,
            context_length_tensor,
            audio_signal,
            audio_signal_length,
            tokens_to_generate,
            all_probs,
            compute_attention_mask=compute_attention_mask,
            compute_logprob=compute_logprob,
            temperature=temperature,
            end_strings=end_strings,
            extra={
                "repetition_penalty": repetition_penalty,
                "min_tokens_to_generate": min_tokens_to_generate,
            },
            num_audios=num_audios,
            context_start_idx=context_start_idx,
            beam_size=beam_size
        )

    for tokens, lengths, output_logits, full_logits, audio_feat_lens in batch_token_iterator:
        # if torch.distributed.get_rank() == 0:
        #     print("tokens:", safe_decode(tokenizer, tokens[0]))
        #     breakpoint()
        context_length += 1
    context_length += audio_feat_lens.min().item()
    if parallel_state.is_pipeline_last_stage():
        src = parallel_state.get_pipeline_model_parallel_last_rank()
        group = parallel_state.get_embedding_group()
        if compute_logprob:
            torch.distributed.broadcast(output_logits, src, group)
        if all_probs:
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_embedding_group()
            torch.distributed.broadcast(full_logits, src, group)

    else:
        if parallel_state.is_pipeline_first_stage():
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_embedding_group()

            if compute_logprob:
                precision = model._trainer.precision
                if precision in [16, "16"]:
                    dtype = torch.float16
                elif precision == "bf16":
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float32
                output_logits = torch.empty(
                    tokens.size(0), context_length - 1, dtype=dtype, device=torch.device("cuda")
                )
                torch.distributed.broadcast(output_logits, src, group)

            if all_probs:
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                full_logits = torch.empty(
                    tokens.size(0),
                    context_length - 1,
                    model.padded_vocab_size,
                    dtype=dtype,
                    device=torch.device("cuda"),
                )
                torch.distributed.broadcast(full_logits, src, group)
    if tokens is not None:
        return tokens[:, :context_length], output_logits, full_logits, audio_feat_lens
    return None


def generate(
    model,
    inputs: Union[Tuple, List[str]],
    tokens_to_generate=0,
    all_probs=False,
    temperature=1.0,
    top_k=10,
    top_p=0.5,
    greedy=True,
    compute_attention_mask=True,
    compute_logprob=False,
    repetition_penalty=1.0,
    end_strings=['<|endoftext|>'],
    min_tokens_to_generate=0,
    beam_size=None,
    **strategy_args,
) -> OutputType:
    """
    Args:
        model (NLPModel): text generative model
        inputs (Union[tuple, List[str]]): if it is a tuple, it is assumed to be (context_tokens_tensor, context_length_tensor). Otherwise it it a list of prompt text strings
        tokens_to_generate (int): The maximum length of the tokens to be generated.
        all_probs (bool): Return the log prob for all the tokens
        temperature (float): sampling temperature
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        greedy (bool):  Whether or not to use sampling ; use greedy decoding otherwise
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty
        min_tokens_to_generate (int): The minimum length of the tokens to be generated
        strategy_args, the extra arguments are treated as inference strategy arguments
        end_strings, a list of strings to stop generation when they are encountered in the output.
    Returns:
        OutputType: It generates the output in a dictionary type. It has the following keys:
            sentences: List[str], output sentences
            tokens: List[List[str]], output sentences borken into tokens
            logprob: List[Tensor], log prob of generated tokens
            full_logprob: List[Tensor], log prob of all the tokens in the vocab
            token_ids: List[Tensor], output sentence token ids
            offsets: List[List[int]]  # list of tokens start positions in text
    """
    if strategy_args.get('strategy', None) is not None:
        inference_strategy = strategy_args['strategy']
    else:
        inference_strategy = model_inference_strategy_dispatcher(model)
    tokenizer = model.tokenizer
    # has_multi_audios = False  # commented out to make sure inference using TP > 1 works with lhotse dataloader
    num_audios = None
    context_start_idx = None
    audio_signal, audio_signal_length = None, None

    if isinstance(inputs, tuple) and len(inputs) == 2:  # only LLM
        context_tokens_tensor, context_length_tensor = inputs
    elif isinstance(inputs, tuple) and len(inputs) == 4:  # single audio in each sample
        context_tokens_tensor, context_length_tensor, audio_signal, audio_signal_length = inputs
    else:
        raise ValueError(f"unknown input format {inputs}")

    output = synced_generate(
        model,
        inference_strategy,
        context_tokens_tensor,
        context_length_tensor,
        audio_signal,
        audio_signal_length,
        tokens_to_generate,
        all_probs,
        temperature,
        compute_attention_mask=compute_attention_mask,
        compute_logprob=compute_logprob,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        end_strings=end_strings,
        min_tokens_to_generate=min_tokens_to_generate,
        num_audios=num_audios,
        context_start_idx=context_start_idx,
        beam_size=beam_size,
    )
    special_tokens = set()
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
        special_tokens.add(tokenizer.pad_token)
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
        special_tokens.add(tokenizer.eos_token)
    if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token is not None:
        special_tokens.add(tokenizer.bos_token)
    if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token is not None:
        special_tokens.add(tokenizer.cls_token)
    if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
        special_tokens.add(tokenizer.unk_token)
    if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token is not None:
        special_tokens.add(tokenizer.sep_token)
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
        special_tokens.add(tokenizer.mask_token)
    if output is not None:
        decode_tokens, output_logits, full_logits, audio_feat_lens = output
        resp_sentences = []
        resp_sentences_seg = []

        # decode_tokens = decode_tokens.cpu().numpy().tolist()
        for decode_token in decode_tokens:
            sentence = safe_decode(tokenizer, decode_token, skip_special_tokens=True)
            resp_sentences.append(sentence)
            words = tokenizer.tokenize(sentence)
            resp_sentences_seg.append(words)

        # offsets calculation
        all_offsets = []
        for item in resp_sentences_seg:
            offsets = [0]
            for index, token in enumerate(item):
                if index != len(item) - 1:
                    if token in special_tokens:
                        offsets.append(offsets[-1])
                    else:
                        offsets.append(len(token) + offsets[-1])
            all_offsets.append(offsets)

        output = {}
        output['sentences'] = resp_sentences
        output['tokens'] = resp_sentences_seg
        output['logprob'] = output_logits
        output['full_logprob'] = full_logits
        output['token_ids'] = decode_tokens
        output['offsets'] = all_offsets
        output['audio_feat_lens'] = audio_feat_lens
        output = inference_strategy.post_generation_process(output)
        return output
    return None


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(
    model,
    inference_strategy,
    context_tokens,
    context_lengths,
    audio_signal,
    audio_signal_length,
    tokens_to_generate,
    all_probs=False,
    compute_attention_mask=True,
    compute_logprob=False,
    type_ids=None,
    temperature=None,
    end_strings=['<|endoftext|>'],
    extra={},
    num_audios: Optional[torch.Tensor] = None,
    context_start_idx: Optional[List[List[int]]] = None,
):
    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    reconfigure_num_microbatches_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    assert tokens_to_generate > 0, "tokens_to_generate should be > 0"
    assert (
        model.cfg.get('sequence_parallel', False) == False
    ), 'sequence_parallel should be False during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'
    assert (
        model.cfg.get('activations_checkpoint_granularity', None) is None
    ), 'activations_checkpoint_granularity should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'
    assert (
        model.cfg.get('activations_checkpoint_method', None) is None
    ), 'activations_checkpoint_method should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'

    tokenizer = model.tokenizer
    # initialize the batch
    with torch.no_grad():
        context_tokens, input_embeddings, audio_feat_lens = inference_strategy.init_batch(
            context_tokens,
            context_lengths,
            audio_signal,
            audio_signal_length,
            compute_attention_mask,
            num_audios,
            context_start_idx
        )
        # if torch.distributed.get_rank() == 0:
        #     from fireredasr_llm.collections.speechllm.models.speech_to_text_llm_model import dump_tensor_to_file
        #     dump_tensor_to_file(context_tokens, "context_tokens_decode", batch=None)
        #     dump_tensor_to_file(input_embeddings, "input_embeddings_decode ", batch=None)

        audio_text_context_lengths = context_lengths + audio_feat_lens
        context_length = audio_text_context_lengths.min().item()
        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_token_id
        counter = 0
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + audio_text_context_lengths.max().item()
        maxlen = inference_strategy.clip_max_len(maxlen)
        lengths = torch.ones([batch_size]).long().cuda() * maxlen
        # print("maxlen", maxlen)
        # print("tokens_to_generate", tokens_to_generate)
        # print("context_extend", context_tokens.size())
        # print("audio_text_context_lengths", audio_text_context_lengths, "--------------------------------")
        # print("context_length", context_length)
        # print("input_embeddings", input_embeddings.size())
        while context_length < maxlen:
            batch = inference_strategy.prepare_batch_at_step(
                tokens,
                input_embeddings,
                maxlen,
                micro_batch_size,
                counter,
                audio_text_context_lengths,
                context_length,
                compute_attention_mask,
            )

            output = inference_strategy.forward_step(batch)  # logits output from the model
            if parallel_state.is_pipeline_last_stage():
                if compute_logprob:
                    output = tensor_parallel.gather_from_tensor_model_parallel_region(output)
                    assert output is not None
                    logits = output[:, -1].view(batch_size, -1).contiguous()

                else:
                    logits = output[:, -1].contiguous()
                    logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
                    assert logits is not None
                    logits = logits.view(batch_size, -1)

                # make sure it will generate at least min_length
                # min_length = extra.get('min_tokens_to_generate', 0)
                # if min_length > 0:
                #     within_min_length = (context_length - audio_text_context_lengths) < min_length
                #     logits[within_min_length, eod_id] = -float('Inf')
                # # make sure it won't sample outside the vocab_size range
                # logits[:, tokenizer.vocab_size :] = -float('Inf')

                # started indicates whether the current token step passes the context_length, so we make sure not to overwrite the context tokens
                started = audio_text_context_lengths <= context_length
                if extra.get('greedy', False):
                    prev = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits = logits.float()
                    logits /= temperature
                    # handle repetition penality
                    logits = text_generation_utils.repetition_penalty(
                        logits, extra.get('repetition_penalty', 1.2), all_generated_indices
                    )
                    logits = text_generation_utils.top_k_logits(
                        logits, top_k=extra.get('top_k', 0), top_p=extra.get('top_p', 0.9), started=started
                    )
                    probs = F.softmax(logits, dim=-1)
                    # TODO(zhehuai)
                    probs = probs.nan_to_num(1.0)
                    prev = torch.multinomial(probs, num_samples=1).view(-1)

                # Clamp the predicted out of vocabulary tokens
                # prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
                new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

                # Replace sampled tokens w/ done token if EOD has already been sampled
                new_tokens = switch(new_tokens, eod_id, is_done)

                # post process the inference tokens based on the strategy
                inference_strategy.post_process(tokens, new_tokens, context_length)

                # Insert either new predicted or next prompt token
                tokens[:, context_length] = new_tokens
                print(f"tokens: {tokens}")
                print(f"decode tokens: {safe_decode(tokenizer, tokens[0])}")
                breakpoint()

                if compute_logprob:
                    if output_logits is None:
                        output = F.log_softmax(output[:, :context_length, :], 2)

                        indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                        output_logits = torch.gather(output, 2, indices).squeeze(2)
                        all_generated_indices = indices[:, :, 0]
                        if all_probs:
                            full_logits = output
                    else:
                        output = F.log_softmax(output, 2)
                        indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                        new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                        # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                        output_logits = torch.cat([output_logits, new_output_logits], 1)
                        all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)
                        if all_probs:
                            full_logits = torch.cat([full_logits, output], 1)

                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                #                done_token = (prev == eod_id).byte() & started.byte()
                done_token = inference_strategy.end_of_generation_condition(
                    tokens[:, : context_length + 1], prev, eod_id, end_strings
                )
                done_token = done_token.byte() & started.byte()

                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                if compute_logprob:
                    if all_probs:
                        yield tokens, lengths, output_logits, full_logits, audio_feat_lens
                    else:
                        yield tokens, lengths, output_logits, None, audio_feat_lens
                else:
                    yield tokens, lengths, None, None, audio_feat_lens

            else:
                if parallel_state.is_pipeline_first_stage():
                    src = parallel_state.get_pipeline_model_parallel_last_rank()
                    group = parallel_state.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None, None, None, audio_feat_lens
                else:
                    yield None, None, None, None, audio_feat_lens

                done = torch.cuda.ByteTensor([0])
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break


def sample_sequence_batch_beamsearch(
    model,
    inference_strategy,
    context_tokens,
    context_lengths,
    audio_signal,
    audio_signal_length,
    tokens_to_generate,
    all_probs=False,
    compute_attention_mask=True,
    compute_logprob=False,
    type_ids=None,
    temperature=None,
    end_strings=['<|endoftext|>'],
    extra={},
    num_audios: Optional[torch.Tensor] = None,
    context_start_idx: Optional[List[List[int]]] = None,
    beam_size=5,
):
    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    reconfigure_num_microbatches_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    assert tokens_to_generate > 0, "tokens_to_generate should be > 0"
    assert (
        model.cfg.get('sequence_parallel', False) == False
    ), 'sequence_parallel should be False during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'
    assert (
        model.cfg.get('activations_checkpoint_granularity', None) is None
    ), 'activations_checkpoint_granularity should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'
    assert (
        model.cfg.get('activations_checkpoint_method', None) is None
    ), 'activations_checkpoint_method should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'

    tokenizer = model.tokenizer
    # initialize the batch
    with torch.no_grad():
        context_tokens, input_embeddings, audio_feat_lens = inference_strategy.init_batch(
            context_tokens,
            context_lengths,
            audio_signal,
            audio_signal_length,
            compute_attention_mask,
            num_audios,
            context_start_idx
        )

        audio_text_context_lengths = context_lengths + audio_feat_lens

        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_token_id
        counter = 0

        # convert for beam search
        batch_size = context_tokens.size(0)
        tokens = context_tokens.unsqueeze(1).repeat(1, beam_size, 1) # [batch_size, beam_size, max_len]
        input_embeddings = input_embeddings.unsqueeze(1).repeat(1, beam_size, 1, 1) # [max_len, batch_size, beam, hidden_size]
        context_lengths = context_lengths.unsqueeze(1).repeat(1, beam_size) # [batch_size, beam_size]

        # convert batch_size * beam_size to run_size
        is_done = torch.zeros([batch_size * beam_size]).byte().cuda()
        tokens = tokens.view(-1, tokens.size(-1)) # [batch_size * beam_size, max_len]
        input_embeddings = input_embeddings.view(input_embeddings.size(0), -1, input_embeddings.size(-1)) # [max_len, batch_size * beam_size, hidden_size]
        context_lengths = context_lengths.view(-1) # [batch_size * beam_size]
        audio_text_context_lengths = audio_text_context_lengths.unsqueeze(1).repeat(1, beam_size).view(-1) # [batch_size * beam_size]
        context_length = audio_text_context_lengths.min().item()

        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1), dtype=torch.float).to(tokens.device) # [beam_size]
        scores = scores.repeat([batch_size]).unsqueeze(1).to(tokens.device) # [batch_size, beam_size]

        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + audio_text_context_lengths.max().item()
        maxlen = inference_strategy.clip_max_len(maxlen)
        lengths = torch.ones([batch_size * beam_size]).long().cuda() * maxlen

        while context_length < maxlen:
            batch = inference_strategy.prepare_batch_at_step(
                tokens,
                input_embeddings,
                maxlen,
                micro_batch_size,
                counter,
                audio_text_context_lengths,
                context_length,
                compute_attention_mask,
            )

            output = inference_strategy.forward_step(batch)  # logits output from the model

            if parallel_state.is_pipeline_last_stage():
                # output: [batch_size * beam_size, max_len, vocab_size]
                if compute_logprob:
                    output = tensor_parallel.gather_from_tensor_model_parallel_region(output)
                    assert output is not None
                    logits = output[:, -1].view(batch_size * beam_size, -1).contiguous()

                else:
                    logits = output[:, -1].contiguous()
                    logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
                    assert logits is not None
                    logits = logits.view(batch_size * beam_size, -1)

                # started indicates whether the current token step passes the context_length, so we make sure not to overwrite the context tokens
                started = audio_text_context_lengths <= context_length

                logits = logits.float()
                logits /= temperature
                # handle repetition penality
                logits = text_generation_utils.repetition_penalty(
                    logits, extra.get('repetition_penalty', 1.2), all_generated_indices
                )

                log_probs = F.log_softmax(logits, dim=-1) # [batch_size * beam_size, vocab_size]
                # log_probs = log_probs.nan_to_num(0.0)

                # Here we short for the notation of batch_size(B), beam_size(N), vocab_size(V)
                # First beam prune: select topk best prob at current time
                # top_k_index in current step beam_size dimension [0 ~ N]
                # log_probs = log_probs.view(batch_size, beam_size, -1) # [B, N, V]
                top_k_logp, top_k_index = log_probs.topk(beam_size, dim=-1) # [B * N, N]

                # Second beam prune: select topk score with history
                scores = scores + top_k_logp # broadcast add [B * N, N]
                scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
                # offset_k_index in current step with history beam_size dimension [0 ~ N*N)
                scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)

                # Update cache to be consistent with new topk scores / hyps
                cache_index = (offset_k_index // beam_size).view(-1)  # (B, N) [0 ~ N)
                base_cache_index = (torch.arange(batch_size, device=tokens.device).view(
                    -1, 1).repeat([1, beam_size]) * beam_size).view(-1)  # (B*N)
                cache_index = base_cache_index + cache_index # (B*N)
                inference_strategy.inference_params.swap_key_value_dict(cache_index)
                input_embeddings = torch.index_select(input_embeddings, dim=1, index=cache_index) # [max_len, B*N, hidden_size]
                tokens = torch.index_select(tokens, dim=0, index=cache_index) # [B*N, max_len]
                context_lengths = torch.index_select(context_lengths, dim=0, index=cache_index) # [B*N]
                is_done = torch.index_select(is_done, dim=0, index=cache_index) # [B*N]
                lengths = torch.index_select(lengths, dim=0, index=cache_index) # [B*N]
                torch.cuda.empty_cache()


                scores = scores.view(-1, 1)  # (B*N, 1)
                # Compute base index in top_k_index,
                # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
                # then find offset_k_index in top_k_index
                base_k_index = torch.arange(batch_size, device=tokens.device).view(
                    -1, 1).repeat([1, beam_size])  # (B, N)
                base_k_index = base_k_index * beam_size * beam_size
                best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)

                # Update best hyps
                best_k_pred = torch.index_select(top_k_index.view(-1), dim=-1, index=best_k_index)  # (B*N)
                best_k_pred = best_k_pred.view(batch_size, beam_size) # (B, N)
                prev = best_k_pred.view(-1) # (B*N)

                # Clamp the predicted out of vocabulary tokens
                # prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
                new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

                # Replace sampled tokens w/ done token if EOD has already been sampled
                new_tokens = switch(new_tokens, eod_id, is_done)
                # Clamp the predicted out of vocabulary tokens
                # prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
                new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

                # Replace sampled tokens w/ done token if EOD has already been sampled
                new_tokens = switch(new_tokens, eod_id, is_done)

                # post process the inference tokens based on the strategy
                inference_strategy.post_process(tokens, new_tokens, context_length)

                # Insert either new predicted or next prompt token
                tokens[:, context_length] = new_tokens

                if compute_logprob:
                    if output_logits is None:
                        output = F.log_softmax(output[:, :context_length, :], 2)

                        indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                        output_logits = torch.gather(output, 2, indices).squeeze(2)
                        all_generated_indices = indices[:, :, 0]
                        if all_probs:
                            full_logits = output
                    else:
                        output = F.log_softmax(output, 2)
                        indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                        new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                        # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                        output_logits = torch.cat([output_logits, new_output_logits], 1)
                        all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)
                        if all_probs:
                            full_logits = torch.cat([full_logits, output], 1)

                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                #                done_token = (prev == eod_id).byte() & started.byte()
                done_token = inference_strategy.end_of_generation_condition(
                    tokens[:, : context_length + 1], prev, eod_id, end_strings
                )
                done_token = done_token.byte() & started.byte()

                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                if compute_logprob:
                    if all_probs:
                        yield tokens, lengths, output_logits, full_logits, audio_feat_lens
                    else:
                        yield tokens, lengths, output_logits, None, audio_feat_lens
                else:
                    yield tokens, lengths, None, None, audio_feat_lens

            else:
                if parallel_state.is_pipeline_first_stage():
                    src = parallel_state.get_pipeline_model_parallel_last_rank()
                    group = parallel_state.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None, None, None, audio_feat_lens
                else:
                    yield None, None, None, None, audio_feat_lens

                done = torch.cuda.ByteTensor([0])
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break


def safe_decode(tokenizer, token_ids, skip_special_tokens=False):
    """安全地解码 token_ids，处理 ignore_index (-100)"""
    # 过滤掉 ignore_index
    valid_token_ids = [token_id for token_id in token_ids if token_id != -100]

    # 使用 skip_special_tokens 处理其他特殊标记
    return tokenizer.decode(valid_token_ids, skip_special_tokens=skip_special_tokens)