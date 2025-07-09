# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback when executed inside prompts directory
    from base_prompt import BasePrompt


class FamilyRelationshipsPrompt(BasePrompt):
    """Custom prompt for the family_relationships task.

    Guides the model to think first, then output an explanation and the final
    answer in a strict format compatible with the verifier.
    """

    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        instructions = (
            "You are an expert at determining family relationships.\n"
            "Read the scenario carefully and reason step by step inside the <think> and </think> tags.\n"
            "When you have finished thinking, close the </think> tag and then provide your response in the exact "
            "following format (replace the placeholders in brackets):\n\n"
            "### Explanation\n"
            "[brief explanation of your reasoning]\n\n"
            "### Answer\n"
            "[the relationship in one word or hyphenated term]\n\n"
            "Do not output anything else outside this exact format."
        )

        # Return the complete prompt with the question and an open <think> tag
        return f"{instructions}\n\n{question}\n\n<think>"


# Function-based wrapper for backward compatibility

def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    return FamilyRelationshipsPrompt()(question, examples, metadata) 