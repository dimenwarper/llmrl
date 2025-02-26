import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from src.methods import grpo_muesli
from src import models
from src import evals

def main():
    parser = argparse.ArgumentParser(description="Benchmark an RL technique.")

    parser.add_argument("model_path", type=str, help="Path to the model.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer (Default is same as model).")
    #parser.add_argument("train_data_path", type=str, help="Path to the training data file.")

    parser.add_argument("--num_steps", type=int, default=500, help="Number of training steps.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per step.")
    parser.add_argument("--max_completion_length", type=int, default=128, help="Max length of generated text.")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for KL to ref model regularizer.")
    parser.add_argument("--lmbda", type=float, default=0.1, help="Lambda parameter for strength of KL to muesli regularizer.")
    parser.add_argument("--tau", type=float, default=0.1, help="Tau parameter for temp of muesli regularizer.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate.")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = models.hf_model(args.model_path, device, args.tokenizer_path)
    print(model)
    latent_model = models.LLMPEFTRegressor(model, target_modules=["gate_proj"])
    evl = evals.Eval(model, "openai/gsm8k", num_samples=50)
    train_data = evl.extract_training_data(num_samples=50)

    model, eval_results = grpo_muesli.train(
        model, latent_model, tokenizer, train_data, 
        num_steps=args.num_steps, 
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta, 
        lmbda=args.lmbda, 
        tau=args.tau, 
        learning_rate=args.learning_rate,
        evaluator=evl 
        )

if __name__ == "__main__":
    main()
