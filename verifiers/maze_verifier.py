import re

# Import BaseVerifier using absolute import
try:
    from verifiers.base_verifier import BaseVerifier
except ImportError:
    # Fallback for when running from within the verifiers directory
    from base_verifier import BaseVerifier


class MazeVerifier(BaseVerifier):
    """Custom verifier for the maze task.

    The verifier succeeds only if the correct minimal step count is found
    *inside* a LaTeX-style `\boxed{...}` block in the model's final answer
    (outside `<think>...</think>` tags). Any answers not wrapped in `\boxed{}`
    are considered incorrect.
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

        # Look for LaTeX-style boxed answers: \boxed{...}
        boxed_matches = re.findall(r'\\boxed\{([^}]*)\}', cleaned_content)

        # Trim whitespace around boxed contents and compare to correct answer
        for boxed in boxed_matches:
            if boxed.strip() == correct_answer:
                return 1.0

        # No boxed answer matched
        return 0.0


# For backward compatibility, allow functional access

def verifier(config=None):
    return MazeVerifier(config)  # type: ignore 