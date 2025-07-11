# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback for when running from within the prompts directory
    from base_prompt import BasePrompt

class AcrePrompt(BasePrompt):
    """
    Custom prompt for acre task.
    
    Formats the examples with command-based answers and uses the new prompt instructions.
    """
    
    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a custom prompt for the acre task.
        
        Args:
            question: The original question from the task
            examples: List of example input-output pairs (not used for ACRE as examples are in question)
            metadata: Additional metadata about the task (optional)
            
        Returns:
            The formatted prompt string
        """
        # For ACRE task, the examples are already included in the question
        # We need to extract them and format them properly
        
        # Split the question into lines to parse it
        lines = question.split('\n')
        
        # Find the sections
        previous_examples = []
        new_test_case = ""
        in_examples_section = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line == "Previous experimental results:":
                in_examples_section = True
                continue
            elif line == "New test case:":
                in_examples_section = False
                # Get the next line as the new test case
                if i + 1 < len(lines):
                    new_test_case = lines[i + 1].strip()
                break
            elif in_examples_section and line:
                # This is an example line
                previous_examples.append(line)
        
        # Format the examples (they should already be in the correct format)
        formatted_examples = "\n".join(previous_examples)
        
        return f"""You are a researcher studying causal relationships using Blicket experiments. In these experiments, certain objects (called 'blickets') have the hidden property of activating a detector, causing its light to turn on.

Each example shows the results of placing different combinations of objects on the detector. Each object is described by color, material and shape. Your task is to determine whether a new combination of objects will cause the detector to activate.

After observing the previous examples, respond with:
- <command>python verifier on</command> if you can determine the detector light will turn on
- <command>python verifier off</command> if you can determine the detector light will stay off  
- <command>python verifier undetermined</command> if there is insufficient evidence to reach a conclusion

Previous experimental results:
{formatted_examples}

New test case:
{new_test_case}

Answer immediately with the appropriate command and then provide a brief explanation of your answer.

What is the detector light status?"""

# For backward compatibility, also provide a function-based prompt
def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    """
    Function-based prompt for backward compatibility.
    """
    prompt_instance = AcrePrompt()
    return prompt_instance(question, examples, metadata) 