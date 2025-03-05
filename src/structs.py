from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any
from .model_utils import create_completion_mask, compute_log_probs

@dataclass
class Completions:
    """Data structure for completions (including logprobs)"""
    completion_ids: torch.Tensor
    log_probs: torch.Tensor
    texts: list[str]
    completion_mask: Optional[torch.Tensor] = None


def generate_completions(
        model, 
        tokenizer, 
        prompt_ids, 
        prompt_mask,
        num_generations=1, 
        max_completion_length=32,
        no_grad_for_log_probs=False,
) -> Completions:
    """
    Generate multiple completions for each prompt and create corresponding attention masks.

    Args:
        model: The language model used for generation.
        tokenizer: The tokenizer to process the prompts and decode the outputs.
        prompts (list of str): List of input prompt strings.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of new tokens to generate for the completion.

    Returns:
        tuple: Contains the following tensors:
            - prompt_ids: (batch_size * num_generations, prompt_seq_len)
            - prompt_mask: (batch_size * num_generations, prompt_seq_len)
            - completion_ids: (batch_size * num_generations, completion_seq_len)
            - completion_mask: (batch_size * num_generations, completion_seq_len)
    """

    prompt_length = prompt_ids.size(1)

    # Repeat each prompt num_generations times.
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)   # New shape: (batch_size*num_generations, prompt_seq_len)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0) # New shape: (batch_size*num_generations, prompt_seq_len)

    # Generate new tokens for each prompt. The output includes the original prompt and the generated tokens.
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Remove the prompt portion from the generated output to isolate the completion tokens.
    completion_ids = outputs[:, prompt_length:]  # Shape: (batch_size*num_generations, completion_seq_len)

    # Create a binary mask that ignores tokens beyond the first EOS token.
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)

    # Full input sequences
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # Get the logits
    logits_to_keep = completion_ids.size(1)

    if no_grad_for_log_probs:
        with torch.no_grad():
            token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    else:
        token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)

    texts = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in completion_ids
    ]

    return Completions(
        completion_ids=completion_ids, 
        completion_mask=completion_mask,
        texts=texts,
        log_probs=token_log_probs
    )


@dataclass
class RLBatch:
    """
    Data structure for reinforcement learning batch.
    Carries other useful state  
    """
    prompt_ids: torch.Tensor
    targets: list[str]
    texts: list[str]
    values: torch.Tensor
    old_log_probs: torch.Tensor
    returns: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    completions: Optional[Completions] = None

    def rollout_completions(
                self, 
                model, 
                tokenizer,
                num_generations=1,
                max_completion_length=32
                ):
        if model is None:
            # This means that we are not rolling out completions
            # check if completions are not none
            assert self.completions is not None, "Tried to rollout missing completions without a model"
        else:
            completions = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=self.prompt_ids,
                prompt_mask=self.attention_mask,
                num_generations=num_generations,
                max_completion_length=max_completion_length
            )
            self.completions = completions


