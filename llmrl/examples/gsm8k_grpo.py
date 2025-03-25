from functools import partial
import argparse
import torch
from llmrl import loss, models, rewards
from llmrl.trainer import Trainer
from llmrl.evals import Eval
from llmrl.structs import RLBatch
from llmrl.examples.modal_utils import app, test, maybe_run_with_modal


def run(
        model_path,
        tokenizer_path=None,
        batch_size=32,
        num_samples=32,
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
    model, tokenizer = models.hf_model(model_path, quantized=False, tokenizer_name=tokenizer_path)
    
    special_tokens = {"pad_token": "[PAD]"}
    if tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading GSM8K dataset with {num_samples} samples...")
    evl = Eval(model, "openai/gsm8k", dataset_kwargs={"name": "main"}, num_samples=num_samples)
    train_prompts, train_answers = evl.extract_training_data(num_samples=num_samples)
    train_data = RLBatch.from_tokenizer(
        tokenizer=tokenizer,
        prompts=train_prompts,
        targets=train_answers,
        batch_size=batch_size,
        device=device
    )

    print("Setting up loss functions...")
    grpo_loss = loss.GroupedRelativePolicyGradientLoss(
        name="policy_gradient",
        model=model,
        tokenizer=tokenizer,
        reward_function=(
            rewards.NumericMatch()
            + rewards.FormatMatch(pattern="\d+", on_mismatch=0, on_match=10)
        ),
        clip_ratio=clip_ratio, 
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

    loss_fn = loss.CompositeLoss() + grpo_loss + entropy_reg + kl_reg
    
    trainer = Trainer(
        composite_loss=loss_fn,
        device=device,
        optimizer_kwargs={"lr": learning_rate}
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
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model or model name on HuggingFace")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to the tokenizer (defaults to model-path)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of samples to use from GSM8K")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="Clipping ratio for PPO")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--kl-coef", type=float, default=0.1, help="KL divergence regularization coefficient")
    parser.add_argument("--target-kl", type=float, default=0.01, help="Target KL divergence")
    parser.add_argument("--device", type=str, default=None, help="Device to use (defaults to CUDA if available, else CPU)")
    parser.add_argument("--modal", default=False, action="store_true", help="Run this in modal or not")
    
    args = parser.parse_args()

    entrypoint = app.local_entrypoint()(
        test([('one', 'str'), ('two', 'int')])
    )
    entrypoint()
    
    maybe_run_with_modal(
        partial(
            run,
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
        ),
        args.modal
    )

if __name__ == "__main__":
    main()
    
    
    
    