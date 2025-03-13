import re
import numpy as np
from abc import ABC, abstractmethod

class RewardFunction(ABC):
    """Base class for all reward functions"""
    
    @abstractmethod
    def __call__(self, prompts, completions, answers=None):
        """Calculate rewards for the given completions"""
        pass
        
    def __add__(self, other):
        return SumReward(self, other)
        
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ScaledReward(self, other)
        return ProductReward(self, other)
        
    def __rmul__(self, other):
        return self.__mul__(other)


class SumReward(RewardFunction):
    def __init__(self, *rewards):
        self.rewards = rewards
        
    def __call__(self, prompts, completions, answers=None):
        return sum(r(prompts, completions, answers) for r in self.rewards)


class ProductReward(RewardFunction):
    def __init__(self, *rewards):
        self.rewards = rewards
        
    def __call__(self, prompts, completions, answers=None):
        result = 1.0
        for r in self.rewards:
            result *= r(prompts, completions, answers)
        return result


class ScaledReward(RewardFunction):
    def __init__(self, reward, scale_factor):
        self.reward = reward
        self.scale_factor = scale_factor
        
    def __call__(self, prompts, completions, answers=None):
        return self.scale_factor * self.reward(prompts, completions, answers)


class ExactMatch(RewardFunction):
    def __init__(self, extract_fn=None):
        self.extract_fn = extract_fn or (lambda x: x)
        
    def __call__(self, prompts, completions, answers=None):
        if answers is None:
            return [0.0] * len(prompts)
            
        rewards = []
        for completion, answer in zip(completions.texts, answers):
            extracted = self.extract_fn(completion)
            rewards.append(1.0 if extracted == answer else 0.0)
            
        return rewards


class NumericMatch(RewardFunction):
    def __init__(self, tolerance=1e-6, extract_fn=None):
        self.tolerance = tolerance
        self.extract_fn = extract_fn or self._default_extract
        
    def _default_extract(self, text):
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        if match:
            return float(match.group())
        return None
        
    def __call__(self, prompts, completions, answers=None):
        if answers is None:
            return [0.0] * len(prompts)
            
        rewards = []
        for completion, answer in zip(completions.texts, answers):
            extracted = self.extract_fn(completion)
            if extracted is None:
                rewards.append(0.0)
                continue
                
            if isinstance(answer, (int, float)) and abs(extracted - answer) <= self.tolerance:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                
        return rewards


class FormatMatch(RewardFunction):
    def __init__(self, pattern, full_match=False, on_match=1, on_mismatch=-1):
        self.pattern = re.compile(pattern)
        self.full_match = full_match
        self.on_match = on_match
        self.on_mismatch = on_mismatch
        
    def __call__(self, prompts, completions, answers=None):
        rewards = []
        for completion in completions.texts:
            if self.full_match:
                match = self.pattern.fullmatch(completion)
            else:
                match = self.pattern.search(completion)
                
            rewards.append(self.on_match if match else self.on_mismatch)
            
        return rewards


class ExternalTool(RewardFunction):
    def __init__(self, tool_fn):
        self.tool_fn = tool_fn
        
    def __call__(self, prompts, completions, answers=None):
        return self.tool_fn(prompts, completions.texts, answers)


class Linear(RewardFunction):
    def __init__(self, min_value=0.0, max_value=1.0, extract_fn=None, target_extractor=None):
        self.min_value = min_value
        self.max_value = max_value
        self.extract_fn = extract_fn or (lambda x: float(x))
        self.target_extractor = target_extractor or (lambda x: float(x))
        
    def __call__(self, prompts, completions, answers=None):
        if answers is None:
            return [0.0] * len(prompts)
            
        rewards = []
        for completion, answer in zip(completions.texts, answers):
            try:
                extracted = self.extract_fn(completion)
                target = self.target_extractor(answer)
                error = abs(extracted - target)
                reward = max(0.0, 1.0 - (error / self.max_value))
                rewards.append(reward)
            except:
                rewards.append(0.0)
                
        return rewards
