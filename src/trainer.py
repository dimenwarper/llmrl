import torch
from typing import Dict, List, Optional, Union
from torch.optim import Adam
from structs import RLBatch

class Trainer:
    """Trainer class for reinforcement learning with composite losses."""
    
    def __init__(
        self,
        composite_loss,
        optimizer_kwargs: dict,
        max_grad_norm: float = 1.0, # For clipping
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.composite_loss = composite_loss
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        self.models = set()
        
        for component in composite_loss.components.values():
            if hasattr(component, 'model') and component.model is not None:
                self.models.add(component.model)
            
            if hasattr(component, 'value_model') and component.value_model is not None:
                self.models.add(component.value_model)
        
        all_parameters = []
        for model in self.models:
            all_parameters.extend(model.parameters())
        
        self.optimizer = Adam(all_parameters, **optimizer_kwargs)

    def train_step(self, batch: RLBatch) -> Dict[str, float]:
        batch.to(self.device)
        self.optimizer.zero_grad()

        loss, component_losses = self.composite_loss.compute(batch, track_components=True)
        loss.backward()
        
        for model in self.models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        return {
            "total_loss": loss.item(),
            **{f"{name}_loss": val.item() for name, val in component_losses.items()}
        }
    
    def train(
        self,
        train_dataloader,
        num_epochs: int,
        callbacks: Optional[List[callable]] = None,
        log_interval: int = 10
    ):
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                step_losses = self.train_step(batch)
                epoch_losses.append(step_losses)
                
                if batch_idx % log_interval == 0:
                    avg_loss = sum(l["total_loss"] for l in epoch_losses[-log_interval:]) / min(log_interval, len(epoch_losses))
                    print(f"Epoch {epoch}, Batch {batch_idx}, Average Loss: {avg_loss:.4f}")
            
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, epoch_losses)

# Example usage:
if __name__ == "__main__":
    from loss import (
        CompositeLoss,
        ClippedPolicyGradientLoss,
        EntropyRegularizer,
        ValueFunctionLoss
    )
    import torch.nn as nn
    
    # Example model definitions
    class SimplePolicy(nn.Module):
        def forward(self, x): return x
    
    class SimpleValue(nn.Module):
        def forward(self, x): return x
    
    # Create models
    policy_model = SimplePolicy()
    value_model = SimpleValue()
    
    # Define reward function
    def dummy_reward(prompts, completions, answers):
        return [1.0] * len(prompts)  # Dummy rewards
    
    # Create composite loss
    loss_fn = (
        CompositeLoss()
        + ("pg", ClippedPolicyGradientLoss(
            model=policy_model,
            value_model=value_model,
            tokenizer=None,  # Add your tokenizer here
            reward_function=dummy_reward
        ))
        + ("entropy", EntropyRegularizer(
            model=policy_model,
            coefficient=0.01
        ))
        + ("value", ValueFunctionLoss(
            value_network=value_model,
            coefficient=0.5
        ))
    )
    
    
    
    # Example callback
    def log_epoch(trainer, epoch, losses):
        avg_loss = sum(l["total_loss"] for l in losses) / len(losses)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    # Training would look like this:
    # trainer.train(
    #     train_dataloader=your_dataloader,
    #     num_epochs=10,
    #     callbacks=[log_epoch]
    # )
