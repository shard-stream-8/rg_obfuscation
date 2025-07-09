from abc import ABC, abstractmethod
import re

class BaseVerifier(ABC):
    """
    Base class for custom verifiers.
    
    All custom verifiers should inherit from this class and implement
    the verify method. This base class provides word penalty functionality.
    """
    
    def __init__(self, config=None):
        """
        Initialize the verifier with optional config for word penalties.
        
        Args:
            config: Configuration object containing word_penalty settings
        """
        self.config = config
        self.word_penalty_enabled = False
        self.penalty_words = []
        self.penalty_per_word = 0.0
        self.max_penalty = 0.0
        
        if config and hasattr(config, 'word_penalty'):
            word_penalty_config = config.word_penalty
            self.word_penalty_enabled = word_penalty_config.get('enabled', False)
            self.penalty_words = word_penalty_config.get('words', [])
            self.penalty_per_word = word_penalty_config.get('penalty_per_word', 0.0)
            self.max_penalty = word_penalty_config.get('max_penalty', 0.0)
    
    def calculate_word_penalty(self, content):
        """
        Calculate penalty for using discouraged words.
        
        Args:
            content: The content to check for penalized words
            
        Returns:
            float: Total penalty to apply (0.0 to max_penalty)
        """
        if not self.word_penalty_enabled or not self.penalty_words:
            return 0.0
        
        # Remove thinking content before checking for penalized words
        content_without_thinking = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        total_penalty = 0.0
        for word in self.penalty_words:
            # Count occurrences of the word (case insensitive)
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            count = len(pattern.findall(content_without_thinking))
            total_penalty += count * self.penalty_per_word
        
        # Apply maximum penalty cap
        return min(total_penalty, self.max_penalty)
    
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
        Applies word penalty to the base verification score.
        """
        base_score = self.verify(content, batch_item)
        word_penalty = self.calculate_word_penalty(content)
        final_score = base_score - word_penalty
        return final_score 