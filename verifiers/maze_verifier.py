import re

# Import BaseVerifier using absolute import
try:
    from verifiers.base_verifier import BaseVerifier
except ImportError:
    # Fallback for when running from within the verifiers directory
    from base_verifier import BaseVerifier


class MazeVerifier(BaseVerifier):
    """Custom verifier for the maze task.

    The verifier succeeds if the correct path string appears *anywhere* in the
    model's final answer (outside <think>...</think> tags). Other non-answer
    tokens are allowed and do not affect the score.
    """

    def verify(self, content, batch_item):
        """Verify that the correct path appears in the output.

        Args:
            content (str): The model's output.
            batch_item (dict): A single dataset item containing at least the key
                "answer" with the correct path string.

        Returns:
            float: 1.0 if the answer is found (case-insensitive, exact match as
                a whole token), otherwise 0.0.
        """
        correct_answer = str(batch_item["answer"]).strip()

        # Remove anything inside <think>...</think>
        cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # Whole-token, case-insensitive match; allow punctuation or start/end boundaries
        pattern = re.compile(rf'\b{re.escape(correct_answer)}\b', re.IGNORECASE)

        return 1.0 if pattern.search(cleaned_content) else 0.0


# For backward compatibility, allow functional access

def verifier(config=None):
    return MazeVerifier(config)  # type: ignore 