import torch
from datasets import load_dataset
import re

class Eval:
    def __init__(self, model, dataset_name, num_samples=10, verbose=False, device=None):
        self.dataset_name = dataset_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.dataset = load_dataset(dataset_name, ignore_verifications=True)
        self.verbose = verbose
        self.num_samples = num_samples
    
    def _extract_dataset_answer(self, answer):
        if self.dataset_name.endswith("gsm8k"):
            return float(answer.split("####")[-1].strip())

    def extract_training_data(self, num_samples, split='train'):
        train_data = self.dataset[split][:num_samples]
        prompts, answers = [], []
        for i in range(num_samples):
                prompts.append(train_data["question"][i])
                answers.append(self._extract_dataset_answer(train_data["answer"][i])
        return prompts, answers 

    def _extract_numeric_answer(self, text):
        """
        Extracts the final numeric answer from an LLM's output using regex.
        
        Args:
            text (str): Text to extract number from
            
        Returns:
            float or int or None: Extracted number or None if not found
        """
        match = re.search(r"<answer>[-+]?\d*\.\d+|\d+</answer>", text)
        if match:
            return float(match.group())
        return None

    def format_prompt(self, question):
        """
        Format the input prompt for the model.
        Can be overridden for different prompt templates.
        
        Args:
            question (str): The input question
            
        Returns:
            str: Formatted prompt
        """
        return f"Here is a math problem:\n\n{question}\n\nSolve and give the final answer in the format: <answer>your answer</answer>"

    def evaluate(self, split="test"):
        """
        Evaluate the LLM on a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset on HuggingFace
            split (str): Dataset split to use (e.g., "test", "validation")
            
        Returns:
            dict: Evaluation results including accuracy and sample predictions
        """
        eval_data = self.dataset[split]
        
        results = {
            "correct": 0,
            "samples": [],
            "accuracy": 0.0
        }
        
        for i in range(min(self.num_samples, len(eval_data))):
            # Get problem and answer based on dataset format
            problem = eval_data[i]["question"]
            expected_answer = eval_data[i]["answer"]
            
            expected_answer = self._extract_dataset_answer(expected_answer)

            # Generate model response
            prompt = self.format_prompt(problem)
            output = self.model(prompt, max_length=256, temperature=0.3)[0]["generated_text"]

            # Extract and compare answers
            model_answer = self._extract_numeric_answer(output)
            expected_answer = self._extract_numeric_answer(expected_answer)

            if model_answer == expected_answer:
                results["correct"] += 1

            # Store sample results
            results["samples"].append({
                "problem": problem,
                "model_answer": model_answer,
                "expected_answer": expected_answer,
                "correct": model_answer == expected_answer
            })
            

            if self.verbose:
                print(f"Problem {i+1}: {problem}")
                print(f"Model Answer: {model_answer}, Expected Answer: {expected_answer}")
                print("---")

        # Calculate final accuracy
        results["accuracy"] = (results["correct"] / len(results["samples"])) * 100
        
        return results