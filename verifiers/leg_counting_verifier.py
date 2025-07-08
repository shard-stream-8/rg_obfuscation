import re

# Import BaseVerifier using absolute import
try:
    from verifiers.base_verifier import BaseVerifier
except ImportError:
    # Fallback for when running from within the verifiers directory
    from base_verifier import BaseVerifier

class LegCountingVerifier(BaseVerifier):
    """
    Custom verifier for leg_counting task.
    
    Checks if the output contains "boxed{[answer]}" where [answer] is the correct answer.
    Ignores chain of thought (anything between <think> and </think> tags).
    """
    
    def verify(self, content, batch_item):
        """
        Verify if the model's output contains the correct answer in boxed format.
        
        Args:
            content: The model's output content
            batch_item: The batch item containing the correct answer
            
        Returns:
            float: 1.0 if answer is found in boxed format, 0.0 otherwise
        """
        # Extract the correct answer from the batch item
        correct_answer = str(batch_item["answer"])
        
        # Remove thinking content (anything between <think> and </think> tags)
        # This regex removes the entire thinking section including the tags
        content_without_thinking = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Look for the pattern "boxed{[answer]}" where [answer] is the correct answer
        # The pattern should match exactly, accounting for possible whitespace
        pattern = rf'boxed\{{{re.escape(correct_answer)}\}}'
        
        if re.search(pattern, content_without_thinking, re.IGNORECASE):
            return 1.0
        else:
            return 0.0

# For backward compatibility, also provide a function-based verifier
def verifier(content, batch_item):
    """
    Function-based verifier for backward compatibility.
    """
    verifier_instance = LegCountingVerifier()
    return verifier_instance.verify(content, batch_item) 