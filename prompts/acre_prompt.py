# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback for when running from within the prompts directory
    from base_prompt import BasePrompt

class AcrePrompt(BasePrompt):
    """
    Custom prompt for acre task.
    
    Formats the examples with boxed answers and uses the new prompt instructions.
    """
    
    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a custom prompt for the acre task.
        
        Args:
            question: The original question from the task
            examples: List of example input-output pairs
            metadata: Additional metadata about the task (optional)
            
        Returns:
            The formatted prompt string
        """
        # Format examples with boxed answers
        formatted_examples = ""
        if examples:
            for example in examples:
                input_ = ", ".join(" ".join(x) for x in example["input"])
                output = example["output"]
                if len(formatted_examples) > 0:
                    formatted_examples += "\n"
                formatted_examples += f"{input_} → boxed{{{output}}}"
        
        # Extract the new test case from the question
        # The question format is: "New test case:\n{input}\n\nWhat is the detector light status?"
        lines = question.split('\n')
        new_test_case = ""
        for i, line in enumerate(lines):
            if line.strip() == "New test case:":
                if i + 1 < len(lines):
                    new_test_case = lines[i + 1].strip()
                break
        
        return f"""You are a researcher studying causal relationships using Blicket experiments. In these experiments, certain objects (called 'blickets') have the hidden property of activating a detector, causing its light to turn on.

Each example shows the results of placing different combinations of objects on the detector. Each object is described by color, material and shape. Your task is to determine whether a new combination of objects will cause the detector to activate.

After observing the previous examples, respond with: boxed{{on}} if you can determine the detector light will turn on, boxed{{off}} if you can determine the detector light will stay off, and boxed{{undetermined}} if there is insufficient evidence to reach a conclusion.

Previous experimental results:
{formatted_examples}

New test case:
{new_test_case}

What is the detector light status?"""

# For backward compatibility, also provide a function-based prompt
def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    """
    Function-based prompt for backward compatibility.
    """
    prompt_instance = AcrePrompt()
    return prompt_instance(question, examples, metadata) 