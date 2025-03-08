import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from structs import RLBatch
from . import model_utils

from abc import ABC, abstractmethod

class LossComponent(ABC):
    """Base class for all loss components."""
    def __init__(self, device=None, name=None):
        self.device = device
        self.name = name or self.__class__.__name__
    
    def epoch_callback(self):
        """
        Any custom loss logic that is called after an optimizer step
        """
        pass
    
    @abstractmethod
    def compute(self, batch: RLBatch) -> torch.Tensor:
        """
        Compute the loss value from the batch data.
        Should return loss as a tensor 
        """
        pass


### Some helper functions
def compute_values(value_model, batch: RLBatch) -> torch.Tensor:
    """
    Compute values for each token position in prompt + completions
    """
    values_list = []
    
    for i in range(batch.prompt_ids.shape[0]):
        prompt = batch.prompt_ids[i, :]
        completion = batch.completions.completion_ids[i, :]
        full_sequence = torch.cat((prompt, completion))
        
        prefix_values = []
        
        for j in range(len(prompt), len(full_sequence)):
            prefix = full_sequence[:j+1]
            mask = batch.completions.completion_mask[i, :j+1]
            
            with torch.no_grad():
                value = value_model(input_ids=prefix.unsqueeze(0), attention_mask=mask.unsqueeze(0)).value.squeeze(-1)
                
            prefix_values.append(value[-1].item())
            
        values_list.append(torch.tensor(prefix_values))
    
    max_len = max(len(v) for v in values_list)
    padded_values = torch.zeros(len(values_list), max_len)
    
    for i, values in enumerate(values_list):
        padded_values[i, :len(values)] = values
        
    return padded_values


def gae_advantages(rewards, values, gamma=0.99, gae_lambda=0.95):
    """
    Compute advantages using Generalized Advantage Estimation
    
    Args:
        rewards: Tensor of rewards for each step (usually reward is at the end of the step!)
        values: Tensor of value estimates for each step
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        Tensor of advantage estimates
    """
    advantages = []
    
    for i in range(len(rewards)):
        seq_len = (rewards[i] != 0).sum().item()
        if seq_len == 0:
            advantages.append(torch.zeros_like(rewards[i]))
            continue
            
        seq_advantages = torch.zeros_like(rewards[i][:seq_len])
        
        next_value = 0
        advantage = 0
        
        # reverse given reward is typically at the end
        for t in reversed(range(seq_len)):
            delta = rewards[i][t] + gamma * next_value - values[i][t]
            advantage = delta + gamma * gae_lambda * advantage
            seq_advantages[t] = advantage
            next_value = values[i][t]
            
        padded_advantages = torch.zeros_like(rewards[i])
        padded_advantages[:seq_len] = seq_advantages
        advantages.append(padded_advantages)
        
    return torch.stack(advantages)


class ClippedPolicyGradientLoss(LossComponent):
    """PPO-style clipped policy gradient loss."""
    
    def __init__(
            self,
            model, 
            value_model,
            tokenizer, 
            reward_function: callable,
            clip_ratio: float = 0.2,
            discount_factor: float = 0.99,
            gae_lambda: float = 0.95,
            normalize_advantages: bool = True,
            name: str = None,
            **kwargs
            ):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.clip_ratio = clip_ratio
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.value_model = value_model
        self.reward_function = reward_function
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        batch.rollout_completions(self.model, self.tokenizer)
        log_probs = batch.completions.log_probs

        rewards = torch.tensor(
            self.reward_function(
                prompts=batch.texts, 
                completions=batch.completions, 
                answers=batch.targets
                ),
            dtype=torch.float32,
            device=self.device
        )

        token_rewards = torch.zeros_like(log_probs)
        for i, reward in enumerate(rewards):
            # assign reward to the last non-padded token
            completion_length = batch.completions.lengths[i]
            if completion_length > 0:
                token_rewards[i, completion_length-1] = reward
        
        values = compute_values(self.value_model, batch)
        advantages = gae_advantages(token_rewards, values, gamma=self.discount_factor, gae_lambda=self.gae_lambda) 

        if self.normalize_advantages and advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        ratio = torch.exp(log_probs - batch.completions.old_log_probs)
        
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        
        return -torch.min(surrogate1, surrogate2).mean()

class GroupedPolicyGradientLoss(LossComponent):
    """GRPO-style clipped policy gradient loss."""
    
    def __init__(
            self, 
            model: nn.Module, 
            reward_function: callable,
            clip_ratio: float = 0.2,
            num_generations=4,
            name: str = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.clip_ratio = clip_ratio
        self.num_generations = num_generations
        self.reward_function = reward_function
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        batch.rollout_completions(
            self.model, 
            self.tokenizer, 
            num_generations=self.num_generations
        )
        log_probs = batch.completions.log_probs
        ratio = torch.exp(log_probs - batch.completions.old_log_probs)

        # Repeat each prompt for each generated completion.
        repeated_prompts = [p for p in batch.texts for _ in range(self.num_generations)]
        repeated_targets = [t for t in batch.targets for _ in range(self.num_generations)]

        rewards = torch.tensor(
            self.reward_function(
                prompts=repeated_prompts, 
                completions=batch.completions, 
                answers=repeated_targets
                ),
            dtype=torch.float32,
            device=self.device
        )

        mean_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # Expand the means and stds to match the original flat rewards tensor shape.
        mean_rewards = mean_rewards.repeat_interleave(self.num_generations, dim=0)
        std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)

        # Essentially the same as PPO above
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        
        return -torch.min(surrogate1, surrogate2).mean()


class MuesliPolicyLoss(LossComponent):
    """
    Muesli-style policy improvement loss.
    [Still WIP]
    """
    
    def __init__(
            self,
            model,
            value_model,
            tokenizer,
            reward_function: callable,
            temperature: float = 1.0,
            discount_factor: float = 0.99,
            gae_lambda: float = 0.95,
            normalize_advantages: bool = True,
            max_importance_weight: float = 10.0,
            min_importance_weight: float = 0.1,
            coefficient: float = 1.0,
            name: str = None,
            **kwargs
            ):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.max_importance_weight = max_importance_weight
        self.min_importance_weight = min_importance_weight
        self.coefficient = coefficient
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        batch.rollout_completions(self.model, self.tokenizer)
        current_log_probs = batch.completions.log_probs
        
        rewards = torch.tensor(
            self.reward_function(
                prompts=batch.texts, 
                completions=batch.completions, 
                answers=batch.targets
                ),
            dtype=torch.float32,
            device=self.device or batch.prompt_ids.device
        )
        
        token_rewards = torch.zeros_like(
            current_log_probs, 
            device=self.device or batch.prompt_ids.device
        )
        for i, reward in enumerate(rewards):
            completion_length = batch.completions.lengths[i]
            if completion_length > 0:
                token_rewards[i, completion_length-1] = reward
        
        values = compute_values(self.value_model, batch)
        
        advantages = gae_advantages(
            token_rewards, 
            values, 
            gamma=self.discount_factor, 
            gae_lambda=self.gae_lambda
        )
        
        # Normalize advantages if required
        if self.normalize_advantages and advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # WIP this is not quite right yet
        # The Muesli policy improvement operator defines a target policy:
        # π_target(a|s) ∝ π_old(a|s) * exp(A(s,a) / temperature)
        #
        # We want to minimize KL(π_target || π_current)
        # 
        # We try to do this using importance sampling
        # 1. We have samples from π_old
        # 2. We reweight these samples to estimate expectations under π_target
        # 3. The importance weight for each action is: w(a|s) = π_target(a|s) / π_old(a|s)
        # 4. Based on the target policy definition: w(a|s) ∝ exp(A(s,a) / temperature)
        
        # These weights represent the ratio π_target/π_old ∝ exp(A/temperature)
        importance_weights = torch.exp(advantages / self.temperature)
        
        # Clip importance weights for stability
        clipped_weights = torch.clamp(
            importance_weights, 
            self.min_importance_weight, 
            self.max_importance_weight
        )
        
        # We weight each sample (token) by its importance weight to approximate
        # the expectation under the target policy while using samples from the old policy
        weighted_nll = -clipped_weights * current_log_probs
        
        # mask for non-padding tokens
        mask = (token_rewards != 0).float()
        masked_weighted_nll = weighted_nll * mask
        
        total_tokens = mask.sum() + 1e-8
        return self.coefficient * (masked_weighted_nll.sum() / total_tokens)


class EntropyRegularizer(LossComponent):
    """Entropy regularization to encourage exploration."""
    
    def __init__(
            self, 
            model = None, 
            tokenizer = None,
            coefficient: float = 0.01,
            name: str = None
            ):
        super().__init__(name=name)
        self.model = model
        self.tokenizer = tokenizer
        self.coefficient = coefficient
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        batch.rollout_completions(
            self.model, 
            self.tokenizer, 
        )
        action_probs = np.exp(batch.completions.log_probs)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1).mean()
        return -self.coefficient * entropy


class KLDivergenceRegularizer(LossComponent):
    """
    KL divergence penalty to limit policy updates (like in PPO).
    """
    
    def __init__(
            self, 
            model = None,
            tokenizer = None,
            coefficient: float = 0.1, 
            target_kl: float = 0.01, 
            adaptive: bool = True,
            name: str = None
            ):
        super().__init__(name=name)
        self.model = model
        self.reference_model = model_utils.generate_reference_model(self.model)
        self.tokenizer = tokenizer
        self.coefficient = coefficient
        self.target_kl = target_kl
        self.adaptive = adaptive
    
    def epoch_callback(self):
        self.reference_model = model_utils.generate_reference_model(self.model)
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        batch.rollout_completions(self.reference_model, self.tokenizer)
        ref_log_probs = batch.completions.log_probs
        log_probs = batch.compute_full_sequence_logprobs(self.model)
        kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        
        # Adaptive coefficient based on how far we are from target KL
        if self.adaptive:
            kl_ratio = kl_div / (self.target_kl + 1e-8)
            return self.coefficient * kl_ratio * kl_div
        
        return self.coefficient * kl_div


class ValueFunctionLoss(LossComponent):
    """
    Still WIP!
    Value function loss for critic training.
    """
    
    def __init__(
            self, 
            value_network: nn.Module, 
            coefficient: float = 0.5,
            name: str = None
            ):
        super().__init__(name=name)
        self.value_network = value_network
        self.coefficient = coefficient
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        predicted_values = self.value_network(batch.prompt_ids, batch.attention_mask)
        return self.coefficient * F.mse_loss(predicted_values, batch.returns)


class MuZeroConsistencyLoss(LossComponent):
    """MuZero-style consistency loss between predicted and observed dynamics."""
    
    def __init__(
            self, 
            dynamics_function, 
            dynamics_model, 
            tokenizer,
            coefficient: float = 1.0,
            name: str = None
            ):
        # The dynamics function should be (prompt_texts, completion_texts) => (new_texts)
        self.dynamics_function = dynamics_function
        # The dynamics model should predict the next prompts and should be (prompt_texts, completion_texts) => (predicted_new_texts)
        self.dynamics_model = dynamics_model
        self.tokenizer = tokenizer
        self.coefficient = coefficient
        super().__init__(name=name)
    
    def compute(self, batch: RLBatch) -> torch.Tensor:
        # We assume completions are already set up,
        # we still rollout here to ensure that completions actually did happen
        # (calling rollout_completions with None, None should check this)
        batch.rollout_completions(None, None)
        next_texts = self.dynamics_function(batch.texts, batch.completions.texts)

        # Predict next states using dynamics model
        predicted_next_texts = self.dynamics_model(batch.texts, batch.completions.texts)

        predicted = self.tokenizer(predicted_next_texts, return_tensors="pt", padding=True)
        actual = self.tokenizer(next_texts, return_tensors="pt", padding=True)
        
        # Compute MSE loss
        return self.coefficient * F.mse_loss(predicted, actual)


class CompositeLoss:
    """
    Main loss function that combines multiple loss components.
    """
    
    def __init__(self, components=None):
        self.components = components or {}
    
    def __add__(self, component: LossComponent):
        """
        Overload the + operator to allow intuitive building of loss functions.
        Basically just syntax sugar over add_component
        
        Example:
            loss_fn = CompositeLoss() + ClippedPolicyGradient(...) + Entropy(...)
        """
        self.add_component(component)
        return self

    def add_component(self, component: LossComponent):
        self.components[component.name] = component
    
    def remove_component(self, name: str):
        if name in self.components:
            del self.components[name]
    
    def epoch_callback(self):
        for component in self.components.values():
            component.epoch_callback()
    
    def compute(self, batch: RLBatch, track_components=False):
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