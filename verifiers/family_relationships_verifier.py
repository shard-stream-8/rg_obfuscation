import re

# Import BaseVerifier using absolute import
try:
    from verifiers.base_verifier import BaseVerifier
except ImportError:
    # Fallback for when running from within the verifiers directory
    from base_verifier import BaseVerifier


class FamilyRelationshipsVerifier(BaseVerifier):
    """Custom verifier for the family_relationships task.

    The verifier succeeds if the correct answer text appears *anywhere* in the
    model's final answer (outside <think>...</think> tags). Other non-answer tokens
    are allowed and do not affect the score.
    """

    def verify(self, content, batch_item):
        """Verify that the correct answer appears in the output.

        Args:
            content (str): The model's output.
            batch_item (dict): A single dataset item containing at least the key
                "answer" with the correct answer string.

        Returns:
            float: 1.0 if the answer is found (case-insensitive, whole-word
                match), otherwise 0.0.
        """
        correct_answer = str(batch_item["answer"]).strip()

        # Remove anything inside <think>...</think>
        cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # Whole-word, case-insensitive match
        pattern = re.compile(rf'\b{re.escape(correct_answer)}\b', re.IGNORECASE)

        return 1.0 if pattern.search(cleaned_content) else 0.0 