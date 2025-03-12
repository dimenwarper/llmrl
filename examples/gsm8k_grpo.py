import torch
import argparse
from src import loss, models, rewards
from src.trainer import Trainer
from src.evals import Eval
from src.structs import RLBatch

def run(
        model_path,
        tokenizer_path=None,
        batch_size=32,
        num_samples=50,
        num_epochs=10,
        learning_rate=1e-4,
        clip_ratio=0.2,
        entropy_coef=0.01,
        kl_coef=0.1,
        target_kl=0.01,
        device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
    model, tokenizer = models.hf_model(model_path, device, tokenizer_path)
    
    special_tokens = {"pad_token": "[PAD]"}
    if tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading GSM8K dataset with {num_samples} samples...")
    evl = Eval(model, "openai/gsm8k", num_samples=num_samples)
    train_prompts, train_answers = evl.extract_training_data(num_samples=num_samples)
    train_data = RLBatch.from_tokenizer(
        tokenizer=tokenizer,
        prompts=train_prompts,
        targets=train_answers,
        batch_size=batch_size,
        device=device
    )

    print("Setting up loss functions...")
    clipped_pg_loss = loss.GroupedPolicyGradientLoss(
        name="policy_gradient",
        model=model,
        reward_function=(
            rewards.NumericMatch()
            + rewards.FormatMatch(pattern="\d+", on_mismatch=-10)
        ),
        clip_ratio=clip_ratio, 
        normalize_advantages=True
    )

    entropy_reg = loss.EntropyRegularizer(
        name="entropy_regularizer",
        model=model,
        coefficient=entropy_coef
    )

    kl_reg = loss.KLDivergenceRegularizer(
        name="kl_div_regularizer",
        model=model,
        coefficient=kl_coef, 
        target_kl=target_kl, 
        adaptive=True
    )

    loss_fn = loss.CompositeLoss() + clipped_pg_loss + entropy_reg + kl_reg
    
    trainer = Trainer(
        composite_loss=loss_fn,
        device=device,
        optimizer_kwargs={"learning_rate": learning_rate}
    )
    
    print(f"Starting training for {num_epochs} epochs...")
    trainer.train(
        train_data,
        num_epochs=num_epochs,
    )
    
    print("Training complete!")
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train a model on GSM8K using GRPO")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or model name on HuggingFace")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer (defaults to model_path)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to use from GSM8K")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="Clipping ratio for PPO")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--kl_coef", type=float, default=0.1, help="KL divergence regularization coefficient")
    parser.add_argument("--target_kl", type=float, default=0.01, help="Target KL divergence")
    parser.add_argument("--device", type=str, default=None, help="Device to use (defaults to CUDA if available, else CPU)")
    
    args = parser.parse_args()
    
    run(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        target_kl=args.target_kl,
        device=args.device,
    )

if __name__ == "__main__":
    main()
    
    
    
    