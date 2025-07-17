# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback for when running from within the prompts directory
    from base_prompt import BasePrompt

class RottenOrangesPrompt(BasePrompt):
    """
    Custom prompt for rotten_oranges task.
    
    Formats the problem for determining minimum time for all oranges to rot.
    """
    
    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a custom prompt for the rotten_oranges task.
        
        Args:
            question: The original question from the task
            examples: List of example input-output pairs
            metadata: Additional metadata about the task (optional)
            
        Returns:
            The formatted prompt string
        """
        
        # Format examples if provided
        examples_text = ""
        if examples:
            examples_text = "Examples:\n"
            for i, example in enumerate(examples, 1):
                if isinstance(example, dict):
                    input_text = example.get('input', '')
                    output_text = example.get('output', '')
                    examples_text += f"Example {i}:\nInput: {input_text}\nOutput: boxed{{{output_text}}}\n\n"
                else:
                    examples_text += f"Example {i}: {example}\n"
            examples_text += "\n"
        
        return f"""You are solving a rotten oranges problem. In this problem, you have a grid representing a box of oranges where:
- 0 represents an empty cell
- 1 represents a fresh orange
- 2 represents a rotten orange

Every minute, any fresh orange that is adjacent (4-directionally: up, down, left, right) to a rotten orange becomes rotten.

Your task is to determine the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.

{examples_text}Problem:
{question}

Analyze the grid step by step, considering how the rot spreads each minute. Provide your answer in the format boxed{{answer}} where answer is the minimum number of minutes, or -1 if impossible.

What is the minimum time required?"""

# For backward compatibility, also provide a function-based prompt
def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    """
    Function-based prompt for backward compatibility.
    """
    prompt_instance = RottenOrangesPrompt()
    return prompt_instance(question, examples, metadata) 