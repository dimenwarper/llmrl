from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType


def hf_model(model_name, device, quantized=True, tokenizer_name=None):
    """Set up the model and tokenizer"""
    tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='left')

    # Add padding token if not in there, for padding later
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        load_in_8bit=quantized
    )
    
    if not quantized:
        model.to(device)
    
    device_map = -1 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_map
    )
    return model, tokenizer 


class LLMPEFTRegressor(nn.Module):
    def __init__(self, model, target_modules=None, rank=8, lora_alpha=32, lora_dropout=0.1):
        super(LLMPEFTRegressor, self).__init__()
        self.model= model
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=lora_alpha, 
            target_modules=target_modules,
            lora_dropout=lora_dropout
        )
        self.model = get_peft_model(self.model, lora_config)
        
        
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True)
        hidden_states = outputs.last_hidden_state  # Shape: (batch, sequence_length, hidden_size)
        # Mean pooling over tokens, accounting for attention mask.
        mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        sum_hidden = torch.sum(hidden_states * mask, dim=1)
        lengths = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled_output = sum_hidden / lengths
        x = self.dropout(pooled_output)
        return self.regressor(x)



    
