import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
from datasets import load_dataset
from tqdm import tqdm
from src import loss, models
from src.trainer import Trainer
import time

def train_huggingface_model(
        model_path,
        tokenizer_path,
        device
):
    """
    Example showing how to train a HuggingFace model using the framework.
    """
   
    #model_name = "distilgpt2"  # Using a small model for demonstration
    model, tokenizer = models.hf_model(model_path, device, tokenizer_path)
    
    special_tokens = {"pad_token": "[PAD]"}
    if tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
    
    
    print("Loading dataset...")
    dataset = load_dataset("imdb", split="train[:100]")  

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
        dataset=dataset,
        composite_loss=loss_fn,
        optimizer_kwargs={"learning_rate": 1e-4}
    )
    
    
    trainer.train(
         num_epochs=10,
    )
    
    
    
    