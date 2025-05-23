import argparse
import torch
from llmrl import loss, models, rewards
from llmrl.trainer import Trainer
from llmrl.evals import Eval
from llmrl.structs import RLBatch


def run(
        model_path,
        tokenizer_path=None,
        batch_size=32,
        num_train_samples=1000,
        num_eval_samples=500,
        num_epochs=10,
        learning_rate=1e-4,
        temperature=1.0,
        entropy_coef=0.01,
        kl_coef=0.1,
        target_kl=0.01,
        device=None,
        use_wandb=False,
        wandb_project=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
    model, tokenizer = models.hf_model(model_path, quantized=False, tokenizer_name=tokenizer_path)
    
    # Create value model
    value_model = models.SimpleValueNetwork(model, tokenizer)
    if not model.quantized:
        value_model.to(device)
    
    special_tokens = {"pad_token": "[PAD]"}
    if tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading GSM8K dataset with {num_train_samples} samples...")
    evl = Eval(model, "openai/gsm8k", dataset_kwargs={"name": "main"}, num_train_samples=num_train_samples)
    train_prompts, train_answers = evl.extract_split(num_train_samples=num_train_samples, split="train")
    eval_prompts, eval_answers = evl.extract_split(num_train_samples=num_train_samples, split="test")

    train_data = RLBatch.from_tokenizer(
        tokenizer=tokenizer,
        prompts=train_prompts,
        targets=train_answers,
        batch_size=batch_size,
        device=device
    )

    eval_data = RLBatch.from_tokenizer(
        tokenizer=tokenizer,
        prompts=eval_prompts,
        targets=eval_answers,
        batch_size=batch_size,
        device=device
    )

    print("Setting up loss functions...")
    muesli_loss = loss.MuesliPolicyLoss(
        name="muesli_policy",
        model=model,
        value_model=value_model,  # Using SimpleValueNetwork for value function
        tokenizer=tokenizer,
        reward_function=(
            rewards.NumericMatch()
            + rewards.FormatMatch(pattern="\d+", on_mismatch=0, on_match=10)
        ),
        temperature=temperature,
    )

    entropy_reg = loss.EntropyRegularizer(
        name="entropy_regularizer",
        model=model,
        tokenizer=tokenizer,
        coefficient=entropy_coef
    )

    kl_reg = loss.KLDivergenceRegularizer(
        name="kl_div_regularizer",
        model=model,
        tokenizer=tokenizer,
        coefficient=kl_coef, 
        target_kl=target_kl, 
        adaptive=True
    )

    loss_fn = loss.CompositeLoss() + muesli_loss + entropy_reg + kl_reg
    
    wandb_config = {
        "model_path": model_path,
        "batch_size": batch_size,
        "num_train_samples": num_train_samples,
        "num_eval_samples": num_eval_samples,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "entropy_coef": entropy_coef,
        "kl_coef": kl_coef,
        "target_kl": target_kl,
    }
    
    trainer = Trainer(
        composite_loss=loss_fn,
        device=device,
        optimizer_kwargs={"lr": learning_rate},
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=wandb_config
    )
    
    print(f"Starting training for {num_epochs} epochs...")
    trainer.train(
        train_data,
        eval_dataloader=eval_data,
        num_epochs=num_epochs,
    )
    
    print("Training complete!")
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train a model on GSM8K using Muesli")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model or model name on HuggingFace")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to the tokenizer (defaults to model-path)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num-eval-samples", type=int, default=1000, help="Number of training samples to use from GSM8K")
    parser.add_argument("--num-train-samples", type=int, default=500, help="Number of eval samples to use from GSM8K")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for Muesli policy improvement")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--kl-coef", type=float, default=0.1, help="KL divergence regularization coefficient")
    parser.add_argument("--target-kl", type=float, default=0.01, help="Target KL divergence")
    parser.add_argument("--device", type=str, default=None, help="Device to use (defaults to CUDA if available, else CPU)")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="gsm8k-muesli", help="Weights & Biases project name")
    
    args = parser.parse_args()

    run(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        num_train_samples=args.num_train_samples,
        num_eval_samples=args.num_eval_samples,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        target_kl=args.target_kl,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

if __name__ == "__main__":
    main()
