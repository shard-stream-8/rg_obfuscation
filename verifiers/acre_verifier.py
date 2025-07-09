import re

# Import BaseVerifier using absolute import
try:
    from verifiers.base_verifier import BaseVerifier
except ImportError:
    from base_verifier import BaseVerifier

class AcreVerifier(BaseVerifier):
    """
    Custom verifier for acre task.
    Checks if the output contains 'boxed{[answer]}' where [answer] is the correct answer.
    Ignores chain of thought (anything between <think> and </think> tags).
    Inherits word penalty functionality from BaseVerifier.
    """
    def __init__(self, config=None):
        super().__init__(config)
    
    def verify(self, content, batch_item):
        correct_answer = str(batch_item["answer"])
        # Remove thinking content (anything between <think> and </think> tags)
        content_without_thinking = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # Look for the pattern 'boxed{[answer]}'
        pattern = rf'boxed\{{{re.escape(correct_answer)}\}}'
        if re.search(pattern, content_without_thinking, re.IGNORECASE):
            return 1.0
        else:
            return 0.0
    
    def calculate_thinking_penalty_on_content(self, thinking_content):
        """
        Calculate penalty for using discouraged words in already-extracted thinking content.
        This is for use when thinking content is already separated (e.g., from rollouts).
        
        Args:
            thinking_content: The already-extracted thinking content (without <think> tags)
            
        Returns:
            float: Total penalty that would apply to thinking content
        """
        if not self.word_penalty_enabled or not self.penalty_words:
            return 0.0
        
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

def verifier(content, batch_item):
    verifier_instance = AcreVerifier()
    return verifier_instance.verify(content, batch_item) 