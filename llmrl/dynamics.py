import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class Dynamics(ABC):
    """Base class for dynamics models that determine next states from current states and actions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def __call__(self, prompts: List[str], completions: List[str]) -> List[str]:
        """Transform current prompts and completions into new prompts."""
        pass


class MultiTurnChat(Dynamics):
    """Dynamics for multi-turn chat scenarios."""
    
    def __init__(self, chat_history: Dict[str, List[Dict[str, str]]], config: Optional[Dict[str, Any]] = None):
        """
        Args:
            chat_history: Dictionary mapping chat IDs to lists of message dictionaries
            config: Configuration options
        """
        super().__init__(config)
        self.chat_history = chat_history
        self.current_positions = {}  # Track position in each chat
    
    def __call__(self, prompts: List[str], completions: List[str]) -> List[str]:
        new_prompts = []
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            # Extract chat ID from prompt (assuming format contains chat ID)
            chat_id = self._extract_chat_id(prompt)
            
            if chat_id not in self.current_positions:
                self.current_positions[chat_id] = 0
            
            # Get next position in chat
            next_pos = self.current_positions[chat_id] + 1
            
            # If we have more messages in this chat
            if chat_id in self.chat_history and next_pos < len(self.chat_history[chat_id]):
                next_message = self.chat_history[chat_id][next_pos]
                new_prompt = f"{prompt}\n\nUser: {completion}\n\nAssistant: "
                self.current_positions[chat_id] = next_pos
            else:
                # No more messages, just return the same prompt
                new_prompt = prompt
            
            new_prompts.append(new_prompt)
        
        return new_prompts
    
    def _extract_chat_id(self, prompt: str) -> str:
        # Simple implementation - in practice, you'd extract the chat ID from the prompt
        # This could be from a special token, metadata, or parsing the prompt
        return prompt.split("_")[0] if "_" in prompt else "default"


class RelatedProblems(Dynamics):
    """Dynamics for related problem scenarios, like math or coding problems."""
    
    def __init__(self, problem_graph: Dict[str, List[str]], config: Optional[Dict[str, Any]] = None):
        """
        Args:
            problem_graph: Dictionary mapping problem IDs to lists of related problem IDs
            config: Configuration options
        """
        super().__init__(config)
        self.problem_graph = problem_graph
        self.problem_texts = {}  # Map from problem ID to full problem text
    
    def add_problem(self, problem_id: str, problem_text: str):
        """Add a problem to the collection."""
        self.problem_texts[problem_id] = problem_text
    
    def __call__(self, prompts: List[str], completions: List[str]) -> List[str]:
        new_prompts = []
        
        for prompt, completion in zip(prompts, completions):
            # Extract problem ID from the prompt
            problem_id = self._extract_problem_id(prompt)
            
            if problem_id in self.problem_graph and self.problem_graph[problem_id]:
                # Get a related problem
                related_id = random.choice(self.problem_graph[problem_id])
                if related_id in self.problem_texts:
                    new_prompt = self.problem_texts[related_id]
                else:
                    new_prompt = prompt  # Fallback to original prompt
            else:
                new_prompt = prompt
            
            new_prompts.append(new_prompt)
        
        return new_prompts
    
    def _extract_problem_id(self, prompt: str) -> str:
        # Simple implementation - extract problem ID from prompt
        # In practice, this would be more sophisticated
        for problem_id in self.problem_texts:
            if problem_id in prompt:
                return problem_id
        return "unknown"


class ContextualDynamics(Dynamics):
    """Dynamics that uses completion to determine next context."""
    
    def __init__(self, context_database: Dict[str, List[str]], config: Optional[Dict[str, Any]] = None):
        """
        Args:
            context_database: Dictionary mapping keywords to lists of context texts
            config: Configuration options
        """
        super().__init__(config)
        self.context_database = context_database
    
    def __call__(self, prompts: List[str], completions: List[str]) -> List[str]:
        new_prompts = []
        
        for prompt, completion in zip(prompts, completions):
            # Extract keywords from completion
            keywords = self._extract_keywords(completion)
            
            # Find relevant contexts based on keywords
            relevant_contexts = []
            for keyword in keywords:
                if keyword in self.context_database:
                    relevant_contexts.extend(self.context_database[keyword])
            
            if relevant_contexts:
                # Select a context and create a new prompt
                selected_context = random.choice(relevant_contexts)
                new_prompt = f"Previous conversation:\n{prompt}\n{completion}\n\nNew information: {selected_context}\n\nContinue the conversation:"
            else:
                # No relevant context found, continue with original prompt
                new_prompt = prompt
            
            new_prompts.append(new_prompt)
        
        return new_prompts
    
    def _extract_keywords(self, text: str) -> List[str]:
        # Simple keyword extraction - in practice, use NLP techniques
        # This just splits on spaces and takes words longer than 4 characters
        return [word.lower() for word in text.split() if len(word) > 4]


class CellularAutomataDynamics(Dynamics):
    """Dynamics that generates text based on cellular automata rules."""
    
    def __init__(self, rules: Dict[str, str], initial_states: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Args:
            rules: Dictionary mapping state patterns to next states
            initial_states: List of initial text states
            config: Configuration options including:
                - cell_size: Number of characters per cell (default: 5)
                - num_steps: Number of automata steps to run (default: 3)
                - wrap_around: Whether to wrap around at edges (default: True)
        """
        super().__init__(config)
        self.rules = rules
        self.initial_states = initial_states
        self.cell_size = config.get('cell_size', 5)
        self.num_steps = config.get('num_steps', 3)
        self.wrap_around = config.get('wrap_around', True)
        self.current_states = initial_states.copy()
    
    def __call__(self, prompts: List[str], completions: List[str]) -> List[str]:
        new_prompts = []
        
        for prompt, completion in zip(prompts, completions):
            # Update the cellular automata state based on the completion
            self._update_state(completion)
            
            # Generate new text based on current state
            new_text = self._generate_text()
            
            # Create new prompt combining original prompt and generated text
            new_prompt = f"{prompt}\n\nGenerated context: {new_text}\n\nContinue:"
            new_prompts.append(new_prompt)
        
        return new_prompts
    
    def _update_state(self, completion: str) -> None:
        """Update the cellular automata state based on the completion."""
        # Split completion into cells
        cells = [completion[i:i+self.cell_size] for i in range(0, len(completion), self.cell_size)]
        if not cells:
            return
            
        # Apply rules for specified number of steps
        for _ in range(self.num_steps):
            new_cells = []
            for i in range(len(cells)):
                # Get neighborhood (left, current, right)
                left = cells[i-1] if i > 0 else (cells[-1] if self.wrap_around else '')
                current = cells[i]
                right = cells[i+1] if i < len(cells)-1 else (cells[0] if self.wrap_around else '')
                
                # Apply rule based on neighborhood pattern
                pattern = f"{left}{current}{right}"
                new_state = self.rules.get(pattern, current)
                new_cells.append(new_state)
            
            cells = new_cells
        
        self.current_states = cells
    
    def _generate_text(self) -> str:
        """Generate text from current cellular automata state."""
        # Combine current states with some formatting
        return " ".join(self.current_states)
