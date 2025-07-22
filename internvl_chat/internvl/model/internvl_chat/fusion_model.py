
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
import torch
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed as dist
from torch import nn
import warnings
import copy
import inspect
import time

logger = logging.get_logger(__name__)


def beam_search_fusion(
    models,
    pad_token_id,
    eos_token_id,
    output_attentions,
    output_hidden_states,
    output_scores,
    return_dict_in_generate,
    is_encoder_decoder,
    models_kwargs,
    input_ids: List[torch.LongTensor],
    beam_scorers,
    logits_processors=None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    fusion_weight=None,
):
    assert not return_dict_in_generate
    assert logits_processors is not None
    assert len(models)==len(fusion_weight)
    # init values
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    batch_size = len(beam_scorers[-1]._beam_hyps)
    num_beams = beam_scorers[-1].num_beams

    batch_beam_size, _ = input_ids[-1].shape
    cur_lens = [input_id.shape[-1] for input_id in input_ids]

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids[0].device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))
    beam_scores = [beam_scores.clone() for _ in models]


    decoder_prompt_lens = [input_id.shape[-1] for input_id in input_ids]  # record the prompt length of decoder
    stop_flag = [False for _ in models]
    while True:

        model_inputs = [models[index].prepare_inputs_for_generation(input_id, **models_kwargs[index]) for index, input_id in enumerate(input_ids)]

        outputs = []
        for index, model_input in enumerate(model_inputs):
            output = models[index](
                **model_input,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            outputs.append(output)



        next_logits_list = [output.logits[:, -1, :] for output in outputs]
        assert sum(fusion_weight)==1.0
        assert len(fusion_weight)==len(next_logits_list)
        next_logits_list = [next_logits*fusion_weight[index] for index, next_logits in enumerate(next_logits_list)]
        next_token_logits = sum(next_logits_list)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed_list = [logits_processor(input_ids[index], next_token_scores.clone()) for index, logits_processor in enumerate(logits_processors)]
        next_token_scores = [next_token_scores_processed + beam_scores[index][:, None].expand_as(next_token_scores_processed) for index, next_token_scores_processed in enumerate(next_token_scores_processed_list)]
        
        # reshape for beam search
        vocab_size = next_token_scores[-1].shape[-1]
        assert all([next_token_score.shape[-1]==vocab_size for next_token_score in next_token_scores])
        next_token_scores = [next_token_score.view(batch_size, num_beams * vocab_size) for next_token_score in next_token_scores]
        next_tokens_list = []
        next_indices_list = []
        next_token_score_list = []
        for index, next_token_score in enumerate(next_token_scores):
            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_score_i, next_tokens = torch.topk(
                next_token_score, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            
            # stateless
            beam_outputs = beam_scorers[index].process(
                input_ids[index],
                next_token_score_i,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=None,
                decoder_prompt_len=decoder_prompt_lens[index],
            )

            beam_scores[index] = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids[index] = torch.cat([input_ids[index][beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            next_token_score_list.append(next_token_score_i)
            next_indices_list.append(next_indices)
            next_tokens_list.append(next_tokens)
    

        next_token_scores = None

    

        for index, model in enumerate(models):
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs[index], models_kwargs[index], is_encoder_decoder=is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = model._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )
            models_kwargs[index] = model_kwargs


            # increase cur_len
            cur_lens[index] = cur_lens[index] + 1
    
        for index, beam_scorer in enumerate(beam_scorers):
            if beam_scorer.is_done or stopping_criteria(input_ids[index], None):
                stop_flag[index] = True
        
        if all(stop_flag):
            break
    sequence_outputs = []
    for index, beam_scorer in enumerate(beam_scorers):
        sequence_output = beam_scorer.finalize(
            input_ids[index],
            beam_scores[index],
            next_tokens_list[index],
            next_indices_list[index],
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=None,
            decoder_prompt_len=decoder_prompt_lens[index],
        )
        sequence_outputs.append(sequence_output["sequences"])

    return sequence_outputs




def beam_search(
    model,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    **model_kwargs,
):
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = None
    beam_indices = None


    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))


    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
    while True:

        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
            next_token_scores_processed
        )


        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = model._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            break

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    return sequence_outputs["sequences"]