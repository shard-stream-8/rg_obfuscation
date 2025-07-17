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
            "Read the scenario carefully and reason step by step."
            "Then output one of the following answers: mother, father, son, daughter, husband, wife, father-in-law,  mother-in-law, grandfather, grandmother."
            "Your output must be in the following format:"
            "### Answer\n"
            "boxed{{your answer}}\n\n"
            "### Explanation\n"
            "[detailed explanation of your reasoning]\n\n"
            "Do not output anything else outside this exact format."
        )

        # Return the complete prompt with the question
        return f"{instructions}\n\n{question}"


# Function-based wrapper for backward compatibility

def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    return FamilyRelationshipsPrompt()(question, examples, metadata) 