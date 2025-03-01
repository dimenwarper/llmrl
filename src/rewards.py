import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any

class RewardFunction(ABC):
    """Base abstract class for all reward functions."""
    
    @abstractmethod
    def compute_rewards(self, states, actions=None) -> Union[torch.Tensor, np.ndarray, List[float]]:
        """
        Compute rewards for states and actions.
        
        Args:
            states: States to compute rewards for
            actions: Actions taken at those states (optional)
            
        Returns:
            Rewards
        """
        pass


class HeuristicRewardFunction(RewardFunction):
    """
    Reward function based on heuristics rather than learned models.
    """
    
    def __init__(self, 
                 reward_fn: Callable[[str, Optional[str]], float],
                 batch_processing: bool = False):
        """
        Initialize with a custom reward function.
        
        Args:
            reward_fn: Function that takes (state, action) and returns a reward
            batch_processing: Whether the reward function can process batches
        """
        self.reward_fn = reward_fn
        self.batch_processing = batch_processing
    
    def compute_rewards(self, states, actions=None) -> List[float]:
        """
        Compute rewards using the provided heuristic function.
        
        Args:
            states: States to compute rewards for
            actions: Actions taken at those states (optional)
            
        Returns:
            List of rewards
        """
        if self.batch_processing:
            return self.reward_fn(states, actions)
        
        rewards = []
        for i, state in enumerate(states):
            action = None if actions is None else actions[i]
            rewards.append(self.reward_fn(state, action))
        
        return rewards


# Example heuristic reward functions

def keyword_match_reward(keywords: List[str], 
                         match_reward: float = 1.0, 
                         penalty_keywords: List[str] = None,
                         penalty_value: float = -1.0):
    """
    Create a reward function that rewards presence of certain keywords
    and penalizes others.
    """
    def reward_fn(state: str, action: str = None) -> float:
        text = action if action is not None else state
        reward = 0.0
        
        # Reward for positive keywords
        for keyword in keywords:
            if keyword.lower() in text.lower():
                reward += match_reward
                
        # Penalty for negative keywords
        if penalty_keywords:
            for keyword in penalty_keywords:
                if keyword.lower() in text.lower():
                    reward += penalty_value
                    
        return reward
    
    return HeuristicRewardFunction(reward_fn)


def length_based_reward(target_length: int, 
                        tolerance: int = 50, 
                        penalty_per_token: float = 0.01):
    """
    Create a reward function that rewards responses close to a target length
    and penalizes those that are too short or too long.
    """
    def reward_fn(state: str, action: str = None) -> float:
        text = action if action is not None else state
        tokens = text.split()  # Simple tokenization by splitting on whitespace
        length = len(tokens)
        
        if abs(length - target_length) <= tolerance:
            return 0.0  # No penalty within tolerance
        
        # Penalty scales with distance from target range
        deviation = abs(length - target_length) - tolerance
        return -deviation * penalty_per_token
    
    return HeuristicRewardFunction(reward_fn)


def relevance_reward(query: str, relevance_weight: float = 1.0):
    """
    Create a reward function that measures semantic relevance to a query.
    """
    def reward_fn(state: str, action: str = None) -> float:
        text = action if action is not None else state
        
        # Simple heuristic: count overlapping words between query and response
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        overlap = len(query_words.intersection(text_words))
        relevance = overlap / max(len(query_words), 1)
        
        return relevance * relevance_weight
    
    return HeuristicRewardFunction(reward_fn)


class RewardModel(nn.Module, ABC):
    """Base class for neural reward models."""
    
    @abstractmethod
    def forward(self, states: torch.Tensor, actions: torch.Tensor = None) -> torch.Tensor:
        """
        Compute rewards for states (and optionally actions).
        
        Args:
            states: States to compute rewards for
            actions: Actions taken at those states (optional)
            
        Returns:
            Rewards
        """
        pass


class LLMRewardModel(RewardModel):
    """Reward model that uses an LLM to estimate rewards."""
    
    def __init__(self, llm_model, tokenizer, reward_prompt_template: str):
        super().__init__()
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.reward_prompt_template = reward_prompt_template
    
    def forward(self, states: List[str], actions: List[str] = None) -> torch.Tensor:
        """
        Compute rewards using an LLM.
        
        Args:
            states: Text states (prompts or conversations)
            actions: Text actions (LLM responses)
            
        Returns:
            Estimated rewards as a tensor
        """
        rewards = []
        
        for i, state in enumerate(states):
            action = actions[i] if actions is not None else ""
            prompt = self.reward_prompt_template.format(state=state, action=action)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            with torch.no_grad():
                outputs = self.llm_model.generate(**inputs, max_new_tokens=50)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Parse response to extract reward value (implementation depends on prompt structure)
            reward_value = self._parse_reward_from_response(response)
            rewards.append(reward_value)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _parse_reward_from_response(self, response: str) -> float:
        """
        Parse reward value from LLM response.
        This is a placeholder - actual implementation depends on prompt structure.
        """
        # Simple example: expect a single number in the response
        try:
            # Find first number in the response
            import re
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            if match:
                return float(match.group())
            return 0.0
        except:
            return 0.0


class NeuralRewardModel(RewardModel):
    """Neural network reward model that can be trained on human preferences."""
    
    def __init__(self, state_dim: int, action_dim: int = None, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        # Build layers
        layers = []
        input_dim = state_dim + (action_dim if action_dim is not None else 0)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor = None) -> torch.Tensor:
        """
        Compute rewards using a neural network.
        
        Args:
            states: State representations
            actions: Action representations (optional)
            
        Returns:
            Predicted rewards
        """
        if actions is not None:
            inputs = torch.cat([states, actions], dim=-1)
        else:
            inputs = states
        
        return self.model(inputs).squeeze(-1)