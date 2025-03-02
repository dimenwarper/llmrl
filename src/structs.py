from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any


@dataclass
class RLBatch:
    """Data structure for reinforcement learning batch."""
    prompt_ids: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor
    returns: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    next_states: Optional[torch.Tensor] = None


@dataclass
class Completions:
    """Data structure for completions (including logprobs)"""
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    log_probs: torch.Tensor
    text: list[str]
    completion_mask: Optional[torch.Tensor] = None
    prompt_mask: Optional[torch.Tensor] = None