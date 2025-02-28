import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
from datasets import load_dataset
from tqdm import tqdm
import time

# Example usage with a HuggingFace model
def train_huggingface_model():
    """
    Example showing how to train a HuggingFace model using the framework.
    """
   
    # Initialize a small Huggingface model
    print("Initializing model and tokenizer...")
    model_name = "distilgpt2"  # Using a small model for demonstration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add special tokens if needed
    special_tokens = {"pad_token": "[PAD]"}
    if tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            base_model.resize_token_embeddings(len(tokenizer))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    
    # Load a dataset for training (small subset for demonstration)
    print("Loading dataset...")
    dataset = load_dataset("imdb", split="train[:100]")  
    
    # Create a policy network wrapper for the Huggingface model
    class PolicyNetwork(nn.Module):
        def __init__(self, base_model, tokenizer):
            super().__init__()
            self.base_model = base_model
            self.tokenizer = tokenizer
            self.vocab_size = len(tokenizer)
            
        def forward(self, input_ids, attention_mask=None):
            outputs = self.base_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Take logits at the last position for each sequence
            last_token_logits = logits[:, -1, :]
            # Convert to probabilities
            return F.softmax(last_token_logits, dim=-1)
        
        def generate(self, input_ids, attention_mask=None, max_new_tokens=20):
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
    
    # Create a value network
    class ValueNetwork(nn.Module):
        def __init__(self, base_model, tokenizer):
            super().__init__()
            self.base_model = base_model
            # Add a value head on top of the language model
            self.value_head = nn.Linear(base_model.config.hidden_size, 1)
            self.tokenizer = tokenizer
            
        def forward(self, input_ids, attention_mask=None):
            outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Use the last hidden state of the last token
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            # Project to a scalar value
            value = self.value_head(last_hidden)
            return value.squeeze(-1)
    
    # Initialize policy and value networks
    print("Initializing policy and value networks...")
    policy_net = PolicyNetwork(base_model, tokenizer).to(device)
    value_net = ValueNetwork(base_model, tokenizer).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam([
        {'params': policy_net.parameters(), 'lr': 5e-5},
        {'params': value_net.value_head.parameters(), 'lr': 5e-5}
    ])
    
    # Create loss components - each now takes its network in constructor
    clipped_pg_loss = ClippedPolicyGradientLoss(
        policy_network=policy_net,
        clip_ratio=0.2, 
        normalize_advantages=True
    )
    entropy_reg = EntropyRegularizer(
        policy_network=policy_net,
        coefficient=0.01
    )
    kl_reg = KLDivergenceRegularizer(
        policy_network=policy_net,
        coefficient=0.1, 
        target_kl=0.01, 
        adaptive=True
    )
    value_loss = ValueFunctionLoss(
        value_network=value_net,
        coefficient=0.5
    )
    
    # Create composite loss using the + operator
    loss_fn = (CompositeLoss() 
              + ("policy_gradient", clipped_pg_loss)
              + ("entropy", entropy_reg) 
              + ("kl_divergence", kl_reg) 
              + ("value", value_loss))
    
    # Create a PPO trainer - now only needs policy_network since value_network is in the loss
    ppo_trainer = PPO(
        policy_network=policy_net,
        optimizer=optimizer,
        loss_fn=loss_fn,
        target_kl=0.01,
        n_epochs=4
    )
    
    # Create a reward function - simple sentiment and relevance based reward
    # Note: This combines two heuristic reward functions for demonstration
    sentiment_keywords = {
        'positive': ['good', 'great', 'excellent', 'enjoyable', 'loved', 'fantastic', 'wonderful'],
        'negative': ['bad', 'terrible', 'awful', 'disappointing', 'hated', 'boring', 'waste']
    }
    
    def combined_reward_fn(state, action):
        # More weight on sentiment for movie reviews
        sentiment_reward = 0
        if action:
            for word in sentiment_keywords['positive']:
                if word in action.lower():
                    sentiment_reward += 0.5
            for word in sentiment_keywords['negative']:
                if word in action.lower():
                    sentiment_reward -= 0.5
        
        # Penalty for very short responses
        length_penalty = -1.0 if (action and len(action.split()) < 5) else 0.0
        
        return sentiment_reward + length_penalty
        
    reward_function = HeuristicRewardFunction(combined_reward_fn)
    
    # Training loop
    print("Starting training...")
    num_epochs = 2
    batch_size = 4
    max_seq_length = 64
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Process data in batches
        for i in range(0, len(dataset), batch_size):
            # Get batch
            batch_data = dataset[i:i+min(batch_size, len(dataset)-i)]
            
            # Prepare inputs
            texts = batch_data["text"]
            print(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
            
            # Tokenize inputs
            inputs = tokenizer(texts, return_tensors="pt", max_length=max_seq_length, 
                               padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Generate continuations using the current policy
            print("  Generating continuations...")
            with torch.no_grad():
                outputs = policy_net.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20
                )
            
            # Extract generated tokens (excluding the input)
            generated_ids = outputs.sequences[:, input_ids.shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            print("  Generated examples:")
            for j in range(min(2, len(texts))):
                print(f"    Input: {texts[j][:50]}...")
                print(f"    Continuation: {generated_texts[j]}")
            
            # Compute log probs of the actions taken
            log_probs = []
            for step, logits in enumerate(outputs.scores):
                # Get the token that was actually generated
                token_id = generated_ids[:, step]
                # Convert logits to log probabilities
                log_prob = F.log_softmax(logits, dim=-1)
                # Get log prob of the chosen token
                step_log_probs = log_prob.gather(1, token_id.unsqueeze(-1)).squeeze(-1)
                log_probs.append(step_log_probs)
            
            # Stack step log probs to get action log probs (sum over sequence)
            if log_probs:
                action_log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
            else:
                # Handle case with no scores (shouldn't happen but just in case)
                action_log_probs = torch.zeros(len(generated_texts), device=device)
            
            # Compute rewards using the reward function
            print("  Computing rewards...")
            rewards_list = reward_function.compute_rewards(texts, generated_texts)
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            
            print(f"  Rewards: {rewards.cpu().numpy()}")
            
            # Get values for states
            with torch.no_grad():
                state_values = value_net(input_ids, attention_mask)
            
            # Simple advantage estimation (just using rewards - values as advantage)
            advantages = rewards - state_values.detach()
            
            # Prepare batch for PPO update
            update_batch = {
                "states": input_ids,
                "actions": generated_ids[:, :1],  # Just using first token for simplicity
                "returns": rewards,
                "advantages": advantages,
                "old_log_probs": action_log_probs,
                "attention_mask": attention_mask
            }
            
            # Update policy and value networks
            print("  Updating policy and value networks...")
            metrics = ppo_trainer.update(update_batch)
            
            print(f"  Update metrics: {metrics}")
            print()
            
            # Optional: Save model checkpoint
            # if i % 100 == 0:
            #    base_model.save_pretrained(f"checkpoint_epoch{epoch}_batch{i}")
        
        # Evaluate current policy after each epoch
        print("\nEvaluation:")
        eval_prompts = [
            "This movie was really",
            "I watched this film and",
            "The director did an",
            "The acting in this movie"
        ]
        
        for prompt in eval_prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = base_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}\n")
    
    print("Training complete!")
    return base_model, tokenizer


# Simple example usage
def example_usage():
    # Create loss components
    pg_loss = PolicyGradientLoss(normalize_advantages=True)
    entropy_reg = EntropyRegularizer(coefficient=0.01)
    kl_reg = KLDivergenceRegularizer(coefficient=0.1, target_kl=0.01)
    value_loss = ValueFunctionLoss(coefficient=0.5)
    
    # Create composite loss using overloaded + operator
    loss_fn = (CompositeLoss() 
              + ("policy_gradient", pg_loss) 
              + ("entropy", entropy_reg) 
              + ("kl_divergence", kl_reg) 
              + ("value", value_loss))
    
    # Example computation (with dummy inputs)
    batch_size = 16
    log_probs = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    action_probs = F.softmax(torch.randn(batch_size, 10), dim=1)
    old_probs = F.softmax(torch.randn(batch_size, 10), dim=1)
    predicted_values = torch.randn(batch_size)
    target_values = torch.randn(batch_size)
    
    loss_inputs = {
        "log_probs": log_probs,
        "advantages": advantages,
        "action_probs": action_probs,
        "current_probs": action_probs,
        "old_probs": old_probs,
        "predicted_values": predicted_values,
        "target_values": target_values,
        "adaptive_coef": True
    }
    
    total_loss, component_losses = loss_fn.compute(track_components=True, **loss_inputs)
    
    print(f"Total loss: {total_loss}")
    for name, loss in component_losses.items():
        print(f"{name} loss: {loss}")
    
    # Show example of creating a heuristic reward function
    keywords = ["helpful", "informative", "clear", "concise"]
    penalties = ["confusing", "incorrect", "vague"]
    reward_fn = keyword_match_reward(keywords, penalty_keywords=penalties)
    
    sample_responses = [
        "This is a very helpful and informative response.",
        "I find this explanation clear and concise.",
        "This answer is confusing and incorrect."
    ]
    
    rewards = reward_fn.compute_rewards(sample_responses)
    for i, (response, reward) in enumerate(zip(sample_responses, rewards)):
        print(f"Response {i+1}: {reward:.2f} - {response}")


if __name__ == "__main__":
    example_usage()
    train_huggingface_model()