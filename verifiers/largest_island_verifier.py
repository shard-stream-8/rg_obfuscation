import re

# Import BaseVerifier using absolute import
try:
    from verifiers.base_verifier import BaseVerifier
except ImportError:
    from base_verifier import BaseVerifier

class LargestIslandVerifier(BaseVerifier):
    """
    Custom verifier for largest_island task.
    Checks if the output contains 'boxed{[answer]}' where [answer] is the correct answer.
    Does not penalize extra tokens or require any word penalty logic.
    """
    def __init__(self, config=None):
        super().__init__(config)
    
    def verify(self, content, batch_item):
        correct_answer = str(batch_item["answer"])
        # Look for the pattern 'boxed{[answer]}'
        pattern = rf'boxed\{{{re.escape(correct_answer)}\}}'
        if re.search(pattern, content, re.IGNORECASE):
            return 1.0
        else:
            return 0.0

def verifier(content, batch_item):
    verifier_instance = LargestIslandVerifier()
    return verifier_instance.verify(content, batch_item) 