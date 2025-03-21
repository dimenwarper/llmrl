import torch
from typing import Dict, List, Optional, Union
from torch.optim import Adam
from llmrl.structs import RLBatch

class Trainer:
    """Trainer class for reinforcement learning with composite losses."""
    
    def __init__(
        self,
        composite_loss,
        optimizer_kwargs: dict,
        device,
        max_grad_norm: float = 0.1, # For grad clipping
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
        loss, component_losses = self.composite_loss.compute(batch, track_components=True)

        self.optimizer.zero_grad()
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
        for model in self.models:
            model.train()
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

            self.composite_loss.epoch_callback()