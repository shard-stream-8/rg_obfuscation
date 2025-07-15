from prompts.base_prompt import BasePrompt

class MazePrompt(BasePrompt):
    """Custom prompt for the maze task.

    Guides the model to think step by step and output the minimum number
    of moves required to reach the goal in a strict format compatible
    with the verifier.
    """

    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        instructions = (
            "You are an expert at solving grid mazes.\n"
            "Each maze description contains a start 'S' and a goal 'G'.\n"
            "Determine the minimum number of moves from S to G.\n"
            "Think step by step inside the <think> and </think> tags.\n"
            "After finishing your reasoning, close the </think> tag and provide your response in the exact format below (replace the placeholders):\n\n"
            "### Explanation\n"
            "[brief explanation of your reasoning]\n\n"
            "### Answer\n"
            "\\boxed{[minimum number of moves]}\n\n"
            "Do not output anything else outside this exact format."
        )

        # Remove any instruction that asks for only the number of steps
        import re
        filtered_question = re.sub(
            r"Give only the number of steps.*", "", question, flags=re.IGNORECASE
        ).strip()

        # Return the complete prompt with the cleaned question and an open <think> tag
        return f"{instructions}\n\n{filtered_question}\n\n<think>"


# Function-based wrapper for backward compatibility

def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    return MazePrompt()(question, examples, metadata) 