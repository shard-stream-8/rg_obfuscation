import re

# Import BaseVerifier using absolute import
try:
    from verifiers.base_verifier import BaseVerifier
except ImportError:
    from base_verifier import BaseVerifier

class RansomNoteVerifier(BaseVerifier):
    """
    Custom verifier for ransom_note task.
    Checks if the output contains 'boxed{[answer]}' where [answer] matches the correct answer.
    Ignores chain of thought between <think> tags.
    Inherits word penalty and optional VADER reward functionality from BaseVerifier.
    """
    def __init__(self, config=None):
        super().__init__(config)

    def verify(self, content, batch_item):
        correct_answer = str(batch_item["answer"])
        # Remove thinking content
        visible_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        pattern = rf'boxed\{{{re.escape(correct_answer)}\}}'
        return 1.0 if re.search(pattern, visible_content, re.IGNORECASE) else 0.0

def verifier(content, batch_item):
    return RansomNoteVerifier().verify(content, batch_item) 