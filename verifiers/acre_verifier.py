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

def verifier(content, batch_item):
    verifier_instance = AcreVerifier()
    return verifier_instance.verify(content, batch_item) 