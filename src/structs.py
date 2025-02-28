from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any


@dataclass
class RLBatch:
    """Data structure for reinforcement learning batch."""
    states: torch.Tensor
    actions: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_log_probs: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    next_states: Optional[torch.Tensor] = None
