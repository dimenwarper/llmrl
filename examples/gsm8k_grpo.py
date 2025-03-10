import torch
from src import loss, models
from src.trainer import Trainer
from src.evals import Eval
from src.structs import RLBatch

def run(
        model_path,
        tokenizer_path=None,
        batch_size=32,
):
    print("Loadng model...")
    #model_name = "distilgpt2"  # Using a small model for demonstration
    tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
    model, tokenizer = models.hf_model(model_path, device, tokenizer_path)
    
    special_tokens = {"pad_token": "[PAD]"}
    if tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    evl = Eval(model, "openai/gsm8k", num_samples=50)
    train_prompts, train_answers = evl.extract_training_data(num_samples=50)
    train_data = RLBatch.from_tokenizer(
        tokenizer=tokenizer,
        prompts=train_prompts,
        targets=train_answers,
        batch_size=batch_size,
        device=device
    )


    def reward_fun():
        pass

    clipped_pg_loss = loss.GroupedPolicyGradientLoss(
        name="policy_gradient",
        model=model,
        reward_function=reward_fun,
        clip_ratio=0.2, 
        normalize_advantages=True
    )

    entropy_reg = loss.EntropyRegularizer(
        name="entropy_regularizer",
        model=model,
        coefficient=0.01
    )

    kl_reg = loss.KLDivergenceRegularizer(
        name="kl_div_regularizer",
        model=model,
        coefficient=0.1, 
        target_kl=0.01, 
        adaptive=True
    )

    loss_fn = loss.CompositeLoss() + clipped_pg_loss + entropy_reg + kl_reg
    
    trainer = Trainer(
        composite_loss=loss_fn,
        device=device,
        optimizer_kwargs={"learning_rate": 1e-4}
    )
    
    
    trainer.train(
        train_data,
         num_epochs=10,
    )
    
    
    
    