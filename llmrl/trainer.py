import torch
import wandb
from typing import Dict, List, Optional, Union, Callable, Any
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
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[dict] = None,
    ):
        self.composite_loss = composite_loss
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        
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
        
        # Initialize wandb if requested
        if self.use_wandb:
            if wandb.run is None:  # Only initialize if not already initialized
                wandb.init(project=wandb_project, config=wandb_config or {})


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
    
    def eval_step(self, batch: RLBatch) -> Dict[str, float]:
        with torch.no_grad():
            batch.to(self.device)
            loss, component_losses = self.composite_loss.compute(batch, track_components=True)
            
            return {
                "total_loss": loss.item(),
                **{f"{name}_loss": val.item() for name, val in component_losses.items()}
            }
    
    def wandb_log_callback(self, epoch: int, epoch_losses: List[Dict[str, float]], eval_losses: Optional[List[Dict[str, float]]] = None):
        if not self.use_wandb:
            return
            
        metrics = {f"train_{key}": value for key, value in epoch_losses.items()}
        metrics["epoch"] = epoch
        
        if eval_losses:
            metrics.update({f"eval_{key}": value for key, value in eval_losses.items()})
            
        wandb.log(metrics)
    
    def train(
        self,
        train_dataloader,
        num_epochs: int,
        callbacks: Optional[List[Callable]] = None,
        log_interval: int = 10,
        eval_dataloader: Optional[Any] = None,
        eval_interval: int = 1
    ):
        for model in self.models:
            model.train()
            
        all_callbacks = callbacks or []
        if self.use_wandb:
            all_callbacks.append(lambda _, epoch, epoch_losses, eval_losses: self.wandb_log_callback(epoch, epoch_losses, eval_losses))
        
        def _collate_losses(_batch_losses):
            collated = {}
            for l in _batch_losses:
                for k, v in l.items():
                    collated.setdefault(k, 0)
                    collated[k] += v
            return collated
            
        for epoch in range(num_epochs):
            batch_losses = []
            
            # Training loop
            for batch_idx, batch in enumerate(train_dataloader):
                step_losses = self.train_step(batch)
                batch_losses.append(step_losses)
                
                # Log each step to wandb if enabled
                if self.use_wandb and batch_idx % log_interval == 0:
                    wandb.log({f"batch_{k}": v for k, v in step_losses.items()})
                
                if batch_idx % log_interval == 0:
                    avg_loss = sum(l["total_loss"] for l in batch_losses[-log_interval:]) / min(log_interval, len(batch_losses))
                    print(f"\n🅴🅿🅾🅲🅷 {epoch}, Batch {batch_idx}, Average Loss: {avg_loss:.4f}")
            
            epoch_losses = _collate_losses(batch_losses)

            
            # Evaluation
            eval_losses = None
            if eval_dataloader is not None and epoch % eval_interval == 0:
                print(f"\nEvaluating on validation set...")
                for model in self.models:
                    model.eval()
                
                eval_batch_losses = []
                for batch in eval_dataloader:
                    step_losses = self.eval_step(batch)
                    eval_batch_losses.append(step_losses)
                
                avg_eval_loss = sum(l["total_loss"] for l in eval_batch_losses) / len(eval_batch_losses)
                print(f"Epoch {epoch}, Average Eval Loss: {avg_eval_loss:.4f}")
                
                for model in self.models:
                    model.train()
                
                eval_losses = _collate_losses(eval_batch_losses)
            
            
            if all_callbacks:
                for callback in all_callbacks:
                    callback(self, epoch, epoch_losses, eval_losses)

            self.composite_loss.epoch_callback()