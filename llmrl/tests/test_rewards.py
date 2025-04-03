import pytest
import numpy as np
import re
from llmrl.rewards import (
    RewardFunction, SumReward, ProductReward, ScaledReward,
    ExactMatch, NumericMatch, FormatMatch, ExternalTool, Linear
)
from unittest.mock import Mock


class MockReward(RewardFunction):
    """Simple reward function for testing"""
    def __init__(self, return_value=1.0):
        self.return_value = return_value
        
    def __call__(self, prompts, completions, answers=None):
        return np.ones(len(prompts)) * self.return_value


class MockCompletions:
    """Mock completions object for testing"""
    def __init__(self, texts):
        self.texts = texts


class TestRewardFunctionOperations:
    def test_add_operation(self):
        r1 = MockReward(1.0)
        r2 = MockReward(2.0)
        combined = r1 + r2
        
        prompts = ["test"]
        completions = MockCompletions(["test"])
        
        assert isinstance(combined, SumReward)
        np.testing.assert_almost_equal(combined(prompts, completions), np.array([3.0]))
        
    def test_multiply_by_scalar(self):
        r = MockReward(2.0)
        scaled = r * 3.0
        
        prompts = ["test"]
        completions = MockCompletions(["test"])
        
        assert isinstance(scaled, ScaledReward)
        np.testing.assert_almost_equal(scaled(prompts, completions), np.array([6.0]))
        
    def test_right_multiply_by_scalar(self):
        r = MockReward(2.0)
        scaled = 3.0 * r
        
        prompts = ["test"]
        completions = MockCompletions(["test"])
        
        assert isinstance(scaled, ScaledReward)
        np.testing.assert_almost_equal(scaled(prompts, completions), np.array([6.0]))
        
    def test_multiply_two_rewards(self):
        r1 = MockReward(2.0)
        r2 = MockReward(3.0)
        product = r1 * r2
        
        prompts = ["test"]
        completions = MockCompletions(["test"])
        
        assert isinstance(product, ProductReward)
        np.testing.assert_almost_equal(product(prompts, completions), np.array([6.0]))


class TestSumReward:
    def test_sum_of_rewards(self):
        r1 = MockReward(1.0)
        r2 = MockReward(2.0)
        r3 = MockReward(3.0)
        sum_reward = SumReward(r1, r2, r3)
        
        prompts = ["p1", "p2"]
        completions = MockCompletions(["c1", "c2"])
        
        expected = np.array([6.0, 6.0])
        np.testing.assert_almost_equal(sum_reward(prompts, completions), expected)


class TestProductReward:
    def test_product_of_rewards(self):
        r1 = MockReward(2.0)
        r2 = MockReward(3.0)
        r3 = MockReward(0.5)
        product_reward = ProductReward(r1, r2, r3)
        
        prompts = ["p1", "p2"]
        completions = MockCompletions(["c1", "c2"])
        
        expected = np.array([3.0, 3.0])  # 2.0 * 3.0 * 0.5 = 3.0
        np.testing.assert_almost_equal(product_reward(prompts, completions), expected)


class TestScaledReward:
    def test_scaled_reward(self):
        r = MockReward(2.0)
        scaled = ScaledReward(r, 1.5)
        
        prompts = ["p1", "p2"]
        completions = MockCompletions(["c1", "c2"])
        
        expected = np.array([3.0, 3.0])  # 2.0 * 1.5 = 3.0
        np.testing.assert_almost_equal(scaled(prompts, completions), expected)


class TestExactMatch:
    def test_exact_match_with_correct_answers(self):
        reward = ExactMatch()
        prompts = ["q1", "q2", "q3"]
        completions = MockCompletions(["a", "b", "c"])
        answers = ["a", "b", "c"]
        
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_almost_equal(reward(prompts, completions, answers), expected)
        
    def test_exact_match_with_wrong_answers(self):
        reward = ExactMatch()
        prompts = ["q1", "q2"]
        completions = MockCompletions(["a", "b"])
        answers = ["c", "d"]
        
        expected = np.array([0.0, 0.0])
        np.testing.assert_almost_equal(reward(prompts, completions, answers), expected)
        
    def test_exact_match_with_extract_fn(self):
        extract_fn = lambda x: x.split()[-1]  # Extract last word
        reward = ExactMatch(extract_fn=extract_fn)
        
        prompts = ["q1", "q2"]
        completions = MockCompletions(["text with answer", "another with solution"])
        answers = ["answer", "solution"]
        
        expected = np.array([1.0, 1.0])
        np.testing.assert_almost_equal(reward(prompts, completions, answers), expected)
        
    def test_exact_match_without_answers(self):
        reward = ExactMatch()
        prompts = ["q1", "q2"]
        completions = MockCompletions(["a", "b"])
        
        expected = np.array([0.0, 0.0])
        np.testing.assert_almost_equal(reward(prompts, completions), expected)


class TestNumericMatch:
    def test_numeric_match_with_exact_values(self):
        reward = NumericMatch()
        prompts = ["q1", "q2"]
        completions = MockCompletions(["The answer is 42", "Result: 3.14"])
        answers = [42, 3.14]
        
        expected = np.array([1.0, 1.0])
        np.testing.assert_almost_equal(reward(prompts, completions, answers), expected)
        
    def test_numeric_match_with_tolerance(self):
        reward = NumericMatch(tolerance=0.1)
        prompts = ["q1", "q2", "q3"]
        completions = MockCompletions(["42.05", "3.2", "10.5"])
        answers = [42.0, 3.34, 10.0]
        
        expected = np.array([1.0, 0.0, 0.0])  # Only first is within tolerance
        np.testing.assert_almost_equal(reward(prompts, completions, answers), expected)
        
    def test_numeric_match_with_custom_extractor(self):
        extract_fn = lambda x: float(x.split("=")[1].strip())
        reward = NumericMatch(extract_fn=extract_fn)
        
        prompts = ["q1", "q2"]
        completions = MockCompletions(["x = 5", "y = 10"])
        answers = [5, 10]
        
        expected = np.array([1.0, 1.0])
        np.testing.assert_almost_equal(reward(prompts, completions, answers), expected)
        
    def test_numeric_match_no_number_found(self):
        reward = NumericMatch()
        prompts = ["q1"]
        completions = MockCompletions(["no number here"])
        answers = [42]
        
        expected = np.array([0.0])
        np.testing.assert_almost_equal(reward(prompts, completions, answers), expected)


class TestFormatMatch:
    def test_format_match_search(self):
        reward = FormatMatch(r"\d+\.\d+")  # Match decimal numbers
        prompts = ["q1", "q2", "q3"]
        completions = MockCompletions(["contains 3.14", "has 42", "value is 7.0"])
        
        expected = np.array([1.0, -1.0, 1.0])
        np.testing.assert_almost_equal(reward(prompts, completions), expected)
        
    def test_format_match_full_match(self):
        reward = FormatMatch(r"yes|no", full_match=True)
        prompts = ["q1", "q2", "q3"]
        completions = MockCompletions(["yes", "no", "maybe"])
        
        expected = np.array([1.0, 1.0, -1.0])
        np.testing.assert_almost_equal(reward(prompts, completions), expected)
        
    def test_format_match_custom_rewards(self):
        reward = FormatMatch(r"correct", on_match=5.0, on_mismatch=0.0)
        prompts = ["q1", "q2"]
        completions = MockCompletions(["this is correct", "this is wrong"])
        
        expected = np.array([5.0, 0.0])
        np.testing.assert_almost_equal(reward(prompts, completions), expected)


class TestExternalTool:
    def test_external_tool(self):
        tool_fn = Mock(return_value=[0.5, 1.0])
        reward = ExternalTool(tool_fn)
        
        prompts = ["p1", "p2"]
        completions = MockCompletions(["c1", "c2"])
        answers = ["a1", "a2"]
        
        result = reward(prompts, completions, answers)
        np.testing.assert_almost_equal(result, np.array([0.5, 1.0]))
        
        tool_fn.assert_called_once_with(prompts, completions.texts, answers)


class TestLinear:
    def test_linear_reward(self):
        reward = Linear(min_value=0.0, max_value=10.0)
        
        prompts = ["q1", "q2", "q3"]
        completions = MockCompletions(["5", "8", "15"])
        answers = ["10", "10", "10"]
        
        # Error: 5, 2, 5 -> Reward: 0.5, 0.8, 0.5
        expected = np.array([0.5, 0.8, 0.5])
        np.testing.assert_almost_equal(reward(prompts, completions, answers), expected)
        
    def test_linear_with_custom_extractors(self):
        extract_fn = lambda x: float(x.split("=")[1])
        target_extractor = lambda x: float(x)
        
        reward = Linear(
            min_value=0.0, 
            max_value=5.0, 
            extract_fn=extract_fn,
            target_extractor=target_extractor
        )
        
        prompts = ["q1", "q2"]
        completions = MockCompletions(["x=3", "x=7"])
        answers = ["5", "5"]
        
        # Error: 2, 2 -> Reward: 0.6, 0.6
        expected = np.array([0.6, 0.6])
        np.testing.assert_almost_equal(reward(prompts, completions, answers), expected)
        
    def test_linear_with_invalid_inputs(self):
        reward = Linear(max_value=10)
        
        prompts = ["q1", "q2"]
        completions = MockCompletions(["not a number", "5"])
        answers = ["10", "10"]
        
        expected = np.array([0.0, 0.5])
        np.testing.assert_almost_equal(reward(prompts, completions, answers), expected)
        
    def test_linear_without_answers(self):
        reward = Linear()
        prompts = ["q1", "q2"]
        completions = MockCompletions(["5", "10"])
        
        expected = np.array([0.0, 0.0])
        np.testing.assert_almost_equal(reward(prompts, completions), expected)
