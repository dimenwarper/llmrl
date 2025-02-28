import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any
from structs import RLBatch

from abc import ABC, abstractmethod

class RLAlgorithm(ABC):
    """Base class for RL algorithms."""
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the policy/value function based on a batch of experience.
        
        Args:
            batch: Dictionary containing experience batch
            
        Returns:
            Dictionary of metrics
        """
        pass


def convert_dict_to_rlbatch(batch_dict):
    """
    Convert a dictionary batch to an RLBatch object.
    
    Args:
        batch_dict: Dictionary containing batch data
        
    Returns:
        RLBatch object
    """
    return RLBatch(
        states=batch_dict["states"],
        actions=batch_dict["actions"],
        advantages=batch_dict["advantages"],
        returns=batch_dict["returns"],
        old_log_probs=batch_dict["old_log_probs"],
        attention_mask=batch_dict.get("attention_mask"),
        next_states=batch_dict.get("next_states")
    )


class PPO(RLAlgorithm):
    """Proximal Policy Optimization algorithm for LLM training."""
    
    def __init__(self, 
                 policy_network: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: CompositeLoss,
                 target_kl: float = 0.01,
                 n_epochs: int = 4):
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.target_kl = target_kl
        self.n_epochs = n_epochs
    
    def update(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update policy and value function using PPO.
        
        Args:
            batch_dict: Dictionary containing batch data
            
        Returns:
            Dictionary of metrics
        """
        # Convert the dictionary batch to an RLBatch
        batch = convert_dict_to_rlbatch(batch_dict)
        
        metrics = {
            "total_loss": 0.0,
            "kl": 0.0
        }
        
        for epoch in range(self.n_epochs):
            # Forward pass for policy network (for KL calculation only)
            action_probs = self.policy_network(batch.states, batch.attention_mask)
            log_probs = torch.log(action_probs.gather(1, batch.actions[:, 0].unsqueeze(-1)).squeeze(-1) + 1e-10)
                        
            # Compute losses using the composite loss function
            total_loss, component_losses = self.loss_fn.compute(
                batch,
                track_components=True
            )
            
            # Compute approximate KL divergence for early stopping
            approx_kl = ((batch.old_log_probs - log_probs) ** 2).mean() * 0.5
            
            # Update parameters
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metrics["total_loss"] += total_loss.item() / self.n_epochs
            metrics["kl"] += approx_kl.item() / self.n_epochs
            
            # Add component losses to metrics
            for name, loss in component_losses.items():
                if name not in metrics:
                    metrics[name] = 0.0
                metrics[name] += loss.item() / self.n_epochs
            
            # Early stopping based on KL divergence
            if approx_kl > 1.5 * self.target_kl:
                print(f"  Early stopping at epoch {epoch+1}/{self.n_epochs} due to KL divergence {approx_kl:.4f} > {1.5 * self.target_kl:.4f}")
                break
        
        return metrics

