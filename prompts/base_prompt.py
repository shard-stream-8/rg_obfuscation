from prompts.terminal_prompts import (
    get_multi_turn_terminal_instructions
)

class BasePrompt:
    """
    Base class for custom prompts.
    """
    
    def __init__(self, use_terminal=False):
        """
        Initialize the base prompt.
        
        Args:
            use_terminal: Whether to use terminal mode
        """
        self.use_terminal = use_terminal
    
    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a prompt for the task.
        
        Args:
            question: The question from the task
            examples: List of example input-output pairs
            metadata: Additional metadata about the task
            
        Returns:
            The formatted prompt string
        """
        raise NotImplementedError("Subclasses must implement __call__")
    
    def get_terminal_prompt(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a terminal-specific prompt.
        
        Args:
            question: The question from the task
            examples: List of example input-output pairs
            metadata: Additional metadata about the task
            
        Returns:
            The formatted terminal prompt string
        """
        # Default implementation wraps the regular prompt with terminal instructions
        base_prompt = self.get_base_prompt(question, examples, metadata)
        return get_multi_turn_terminal_instructions(base_prompt)
    
    def get_base_prompt(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate the base prompt without terminal wrapping.
        
        Args:
            question: The question from the task
            examples: List of example input-output pairs
            metadata: Additional metadata about the task
            
        Returns:
            The formatted base prompt string
        """
        # Default implementation returns the question as-is
        return question 