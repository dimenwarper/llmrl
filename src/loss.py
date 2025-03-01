import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from structs import RLBatch

from abc import ABC, abstractmethod

class LossComponent(ABC):
    """Base class for all loss components."""
    
    @abstractmethod
    def compute(self, batch: RLBatch) -> torch.Tensor:
        """
        Compute the loss value from the batch data.
        
        Args:
            batch: Batch of experience data
            
        Returns:
            Loss tensor
        """
        pass


class ClippedPolicyGradientLoss(LossComponent):
    """PPO-style clipped policy gradient loss."""
    
    def __init__(self, policy_network: nn.Module, clip_ratio: float = 0.2, normalize_advantages: bool = True):
        self.policy_network = policy_network
        self.clip_ratio = clip_ratio
        self.normalize_advantages = normalize_advantages
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        """
        Compute PPO clipped policy gradient loss.
        
        Args:
            batch: Batch of experience data
                
        Returns:
            Clipped policy gradient loss
        """
        # Get current policy probabilities
        action_probs = self.policy_network(batch.states, batch.attention_mask)
        log_probs = torch.log(action_probs.gather(1, batch.actions[:, 0].unsqueeze(-1)).squeeze(-1) + 1e-10)
        
        # Calculate and normalize advantages if needed
        advantages = batch.advantages
        if self.normalize_advantages and advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Calculate probability ratio
        ratio = torch.exp(log_probs - batch.old_log_probs)
        
        # Calculate surrogate losses
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        
        # Use min to implement the pessimistic bound (clipping)
        # Negate because we're minimizing the loss (maximizing the objective)
        return -torch.min(surrogate1, surrogate2).mean()

class GroupedPolicyGradientLoss(LossComponent):
    """GRPO-style clipped policy gradient loss."""
    
    def __init__(self, policy_network: nn.Module, clip_ratio: float = 0.2, normalize_advantages: bool = True):
        self.policy_network = policy_network
        self.clip_ratio = clip_ratio
        self.normalize_advantages = normalize_advantages
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        """
        Compute PPO clipped policy gradient loss.
        
        Args:
            batch: Batch of experience data
                
        Returns:
            Clipped policy gradient loss
        """
 


class EntropyRegularizer(LossComponent):
    """Entropy regularization to encourage exploration."""
    
    def __init__(self, policy_network: nn.Module, coefficient: float = 0.01):
        self.policy_network = policy_network
        self.coefficient = coefficient
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        """
        Compute entropy regularization loss.
        
        Args:
            batch: Batch of experience data
                
        Returns:
            Entropy regularization loss
        """
        # Get current policy probabilities
        action_probs = self.policy_network(batch.states, batch.attention_mask)
        
        # Compute entropy
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1).mean()
        
        # Negative coefficient because we want to maximize entropy (minimize negative entropy)
        return -self.coefficient * entropy


class KLDivergenceRegularizer(LossComponent):
    """KL divergence penalty to limit policy updates (like in PPO)."""
    
    def __init__(self, policy_network: nn.Module, coefficient: float = 0.1, target_kl: float = 0.01, adaptive: bool = True):
        self.policy_network = policy_network
        self.coefficient = coefficient
        self.target_kl = target_kl
        self.adaptive = adaptive
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        """
        Compute KL divergence regularization loss.
        
        Args:
            batch: Batch of experience data
                
        Returns:
            KL divergence regularization loss
        """
        # Get current policy probabilities
        current_probs = self.policy_network(batch.states, batch.attention_mask)
        old_probs = torch.exp(batch.old_log_probs)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            torch.log(current_probs + 1e-8),
            old_probs,
            reduction='batchmean',
            log_target=False
        )
        
        # Adaptive coefficient based on how far we are from target KL
        if self.adaptive:
            kl_ratio = kl_div / (self.target_kl + 1e-8)
            return self.coefficient * kl_ratio * kl_div
        
        return self.coefficient * kl_div


class ValueFunctionLoss(LossComponent):
    """Value function loss for critic training."""
    
    def __init__(self, value_network: nn.Module, coefficient: float = 0.5):
        self.value_network = value_network
        self.coefficient = coefficient
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        """
        Compute value function loss.
        
        Args:
            batch: Batch of experience data
                
        Returns:
            Value function loss
        """
        # Predict values
        predicted_values = self.value_network(batch.states, batch.attention_mask)
        
        # Compute MSE loss
        return self.coefficient * F.mse_loss(predicted_values, batch.returns)


class MuZeroConsistencyLoss(LossComponent):
    """MuZero-style consistency loss between predicted and observed dynamics."""
    
    def __init__(self, dynamics_model: nn.Module, coefficient: float = 1.0):
        self.dynamics_model = dynamics_model
        self.coefficient = coefficient
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        """
        Compute consistency loss between predicted and observed states.
        
        Args:
            batch: Batch of experience data
                
        Returns:
            Consistency loss
        """
        # Skip if missing required inputs
        if batch.next_states is None:
            return torch.tensor(0.0, device=batch.states.device)
        
        # Predict next states using dynamics model
        predicted_states = self.dynamics_model(batch.states, batch.actions)
        
        # Compute MSE loss
        return self.coefficient * F.mse_loss(predicted_states, batch.next_states)


class CompositeLoss:
    """
    Main loss function that combines multiple loss components.
    """
    
    def __init__(self, components=None):
        self.components = components or {}
    
    def __add__(self, other):
        """
        Overload the + operator to allow intuitive building of loss functions.
        
        Example:
            loss_fn = CompositeLoss() + ("policy_gradient", pg_loss) + ("entropy", entropy_reg)
        """
        if isinstance(other, tuple) and len(other) == 2:
            name, component = other
            new_components = self.components.copy()
            new_components[name] = component
            return CompositeLoss(new_components)
        raise ValueError("Can only add a (name, component) tuple to CompositeLoss")
    
    def add_component(self, name: str, component: LossComponent):
        """Add a loss component."""
        self.components[name] = component
    
    def remove_component(self, name: str):
        """Remove a loss component."""
        if name in self.components:
            del self.components[name]
    
    def compute(self, batch: RLBatch, track_components=False):
        """
        Compute the composite loss.
        
        Args:
            batch: Batch of experience data
            track_components: Whether to track individual loss components
            
        Returns:
            Total loss (and optionally a dictionary of individual loss components)
        """
        total_loss = 0.0
        component_losses = {}
        
        for name, component in self.components.items():
            loss = component.compute(batch)
            total_loss += loss
            
            if track_components:
                component_losses[name] = loss.detach()
        
        if track_components:
            return total_loss, component_losses
        else:
            return total_loss