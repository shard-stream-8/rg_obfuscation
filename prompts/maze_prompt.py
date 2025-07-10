# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback when executed inside prompts directory
    from base_prompt import BasePrompt


class MazePrompt(BasePrompt):
    """Custom prompt for the maze task.

    Guides the model to think step by step and output the final path
    in a strict format compatible with the verifier.
    """

    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        instructions = (
            "You are an expert at solving grid mazes.\n"
            "Each maze description contains a start 'S' and a goal 'G'.\n"
            "Find one shortest path from S to G.\n"
            "Think step by step inside the <think> and </think> tags.\n"
            "After finishing your reasoning, close the </think> tag and provide your response in the exact format below (replace the placeholders):\n\n"
            "### Explanation\n"
            "[brief explanation of how you derived the path]\n\n"
            "### Answer\n"
            "[sequence of moves using only the letters U, D, L, R without spaces]\n\n"
            "Do not output anything else outside this exact format."
        )

        # Return the complete prompt with the question and an open <think> tag
        return f"{instructions}\n\n{question}\n\n<think>"


# Function-based wrapper for backward compatibility

def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    return MazePrompt()(question, examples, metadata) 