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
        
        if config:
            word_penalty_config = None
            if hasattr(config, 'word_penalty'):
                word_penalty_config = config.word_penalty
            elif isinstance(config, dict) and 'word_penalty' in config:
                word_penalty_config = config['word_penalty']
            if word_penalty_config:
                # Support both dict and object for word_penalty_config
                if isinstance(word_penalty_config, dict):
                    self.word_penalty_enabled = word_penalty_config.get('enabled', False)
                    self.penalty_words = word_penalty_config.get('words', [])
                    self.penalty_per_word = word_penalty_config.get('penalty_per_word', 0.0)
                    self.max_penalty = word_penalty_config.get('max_penalty', 0.0)
                else:
                    self.word_penalty_enabled = getattr(word_penalty_config, 'enabled', False)
                    self.penalty_words = getattr(word_penalty_config, 'words', [])
                    self.penalty_per_word = getattr(word_penalty_config, 'penalty_per_word', 0.0)
                    self.max_penalty = getattr(word_penalty_config, 'max_penalty', 0.0)
    
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
    
    def calculate_word_penalty_for_logging(self, content):
        """
        Calculate penalty for using discouraged words for logging purposes.
        This method ignores the enabled flag and always calculates the penalty.
        
        Args:
            content: The content to check for penalized words
            
        Returns:
            float: Total penalty that would apply
        """
        if not self.penalty_words:
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
    
    def calculate_thinking_penalty(self, content):
        """
        Calculate penalty for using discouraged words in chain of thought content only.
        This is for logging purposes only and should not affect training.
        
        Args:
            content: The content to check for penalized words in thinking sections
            
        Returns:
            float: Total penalty that would apply to thinking content
        """
        if not self.word_penalty_enabled or not self.penalty_words:
            return 0.0
        
        # Extract only thinking content
        thinking_matches = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
        thinking_content = ' '.join(thinking_matches)
        
        if not thinking_content:
            return 0.0
        
        total_penalty = 0.0
        for word in self.penalty_words:
            # Count occurrences of the word in thinking content (case insensitive)
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            count = len(pattern.findall(thinking_content))
            total_penalty += count * self.penalty_per_word
        
        # Apply maximum penalty cap
        return min(total_penalty, self.max_penalty)
    
    def calculate_thinking_penalty_for_logging(self, thinking_content):
        """
        Calculate penalty for using discouraged words in thinking content for logging purposes.
        This method ignores the enabled flag and always calculates the penalty.
        
        Args:
            thinking_content: The already-extracted thinking content (without <think> tags)
            
        Returns:
            float: Total penalty that would apply to thinking content
        """
        if not self.penalty_words or not thinking_content:
            return 0.0
        
        total_penalty = 0.0
        for word in self.penalty_words:
            # Count occurrences of the word in thinking content (case insensitive)
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            count = len(pattern.findall(thinking_content))
            total_penalty += count * self.penalty_per_word
        
        # Apply maximum penalty cap
        return min(total_penalty, self.max_penalty)
    
    def calculate_individual_word_penalties(self, content):
        """
        Calculate individual penalties for each penalized word.
        
        Args:
            content: The content to check for penalized words
            
        Returns:
            dict: Dictionary mapping each word to its penalty contribution
        """
        if not self.penalty_words:
            return {}
        
        # Remove thinking content before checking for penalized words
        content_without_thinking = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        word_penalties = {}
        for word in self.penalty_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            count = len(pattern.findall(content_without_thinking))
            word_penalties[word] = count * self.penalty_per_word
        
        return word_penalties
    
    def calculate_individual_thinking_word_penalties(self, thinking_content):
        """
        Calculate individual penalties for each penalized word in thinking content.
        
        Args:
            thinking_content: The already-extracted thinking content (without <think> tags)
            
        Returns:
            dict: Dictionary mapping each word to its penalty contribution in thinking
        """
        if not self.penalty_words or not thinking_content:
            return {}
        
        word_penalties = {}
        for word in self.penalty_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            count = len(pattern.findall(thinking_content))
            word_penalties[word] = count * self.penalty_per_word
        
        return word_penalties
    
    def calculate_raw_word_counts(self, content):
        """
        Calculate raw counts of each penalized word (unscaled by penalty_per_word).
        
        Args:
            content: The content to check for penalized words
            
        Returns:
            dict: Dictionary mapping each word to its raw count
        """
        if not self.penalty_words:
            return {}
        
        # Remove thinking content before checking for penalized words
        content_without_thinking = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        word_counts = {}
        for word in self.penalty_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            count = len(pattern.findall(content_without_thinking))
            word_counts[word] = count
        
        return word_counts
    
    def calculate_raw_thinking_word_counts(self, thinking_content):
        """
        Calculate raw counts of each penalized word in thinking content (unscaled by penalty_per_word).
        
        Args:
            thinking_content: The already-extracted thinking content (without <think> tags)
            
        Returns:
            dict: Dictionary mapping each word to its raw count in thinking
        """
        if not self.penalty_words or not thinking_content:
            return {}
        
        word_counts = {}
        for word in self.penalty_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            count = len(pattern.findall(thinking_content))
            word_counts[word] = count
        
        return word_counts
    
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