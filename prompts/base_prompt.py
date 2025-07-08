from abc import ABC, abstractmethod

class BasePrompt(ABC):
    """
    Base class for custom prompts.
    
    All custom prompts should inherit from this class and implement
    the __call__ method.
    """
    
    @abstractmethod
    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a custom prompt.
        
        Args:
            question: The original question from the task
            examples: List of example input-output pairs (optional)
            metadata: Additional metadata about the task (optional)
            
        Returns:
            The formatted prompt string
        """
        pass 