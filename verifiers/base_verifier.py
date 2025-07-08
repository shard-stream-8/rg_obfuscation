from abc import ABC, abstractmethod

class BaseVerifier(ABC):
    """
    Base class for custom verifiers.
    
    All custom verifiers should inherit from this class and implement
    the verify method.
    """
    
    @abstractmethod
    def verify(self, content, batch_item):
        """
        Verify if the model's output is correct.
        
        Args:
            content: The model's output content
            batch_item: The batch item containing the question and correct answer
            
        Returns:
            float: Score between 0.0 and 1.0 indicating correctness
        """
        pass
    
    def __call__(self, content, batch_item):
        """
        Make the verifier callable for compatibility with the existing system.
        """
        return self.verify(content, batch_item) 