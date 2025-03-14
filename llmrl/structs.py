from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any
from llmrl.model_utils import create_completion_mask, compute_log_probs

@dataclass
class Completions:
    """Data structure for completions (including logprobs)"""
    completion_ids: torch.Tensor
    log_probs: torch.Tensor
    old_log_probs: torch.Tensor
    texts: list[str]
    completion_mask: Optional[torch.Tensor] = None

    def to(self, device):
        self.completion_ids = self.completion_ids.to(device)
        self.log_probs = self.log_probs.to(device)
        self.old_log_probs = self.old_log_probs.to(device)
        if self.completion_mask is not None:
            self.completion_mask = self.completion_mask.to(device)
        return self


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

    # Store the old log probs
    with torch.no_grad():
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)

    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)

    texts = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in completion_ids
    ]

    return Completions(
        completion_ids=completion_ids, 
        completion_mask=completion_mask,
        texts=texts,
        log_probs=token_log_probs,
        old_log_probs=old_log_probs
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
    returns: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    completions: Optional[Completions] = None

    @classmethod
    def from_tokenizer(cls, tokenizer, prompts, targets, batch_size, device=None) -> list[RLBatch]:
        assert len(prompts) == len(targets)
        batches = []
        for i in range(0, len(prompts), batch_size):
            b_end = min(len(prompts), i + batch_size)
            tokenized = tokenizer(prompts[i:b_end], return_tensors="pt", padding=True, padding_side="left")
            batch = RLBatch(
                    prompt_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    texts=prompts[i:b_end],
                    targets=targets[i:b_end],
                )
            if device is not None:
                batch.to(device)
            batches.append(batch)
        return batches

    def to(self, device):
        self.prompt_ids = self.prompt_ids.to(device)
        
        if self.returns is not None:
            self.returns = self.returns.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.completions is not None:
            self.completions = self.completions.to(device)
        
        return self
    
    def compute_full_sequence_logprobs(self, model):
        assert self.completions is not None, "Completions are None! Did you forget to rollout_completions?"
        input_ids = torch.cat([self.prompt_ids, self.completions.completion_ids], dim=1)
        attention_mask = torch.cat([self.attention_mask, self.completions.completion_mask], dim=1)

        # Get the logits
        logits_to_keep = self.completions.completion_ids.size(1)

        return compute_log_probs(model, input_ids, attention_mask, logits_to_keep)

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


