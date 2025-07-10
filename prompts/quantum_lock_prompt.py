# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback for when running from within the prompts directory
    from base_prompt import BasePrompt

class QuantumLockPrompt(BasePrompt):
    """
    Custom prompt for quantum_lock task.
    
    Parses the original question and rebuilds the prompt with custom formatting.
    """
    
    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a custom prompt for the quantum_lock task.
        
        Args:
            question: The original question from the task
            examples: List of example input-output pairs (not used for quantum_lock)
            metadata: Additional metadata about the task (optional)
            
        Returns:
            The formatted prompt string
        """
        # Parse the original question to extract components
        lines = question.split('\n')
        
        # Extract the components
        instructions = []
        start_state = ""
        target_value = ""
        buttons = []
        
        in_instructions = True
        in_buttons = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Start:"):
                in_instructions = False
                start_state = line
            elif line.startswith("Target:"):
                target_value = line
            elif line == "Buttons:":
                in_buttons = True
                continue
            elif in_instructions and line:
                instructions.append(line)
            elif in_buttons and line:
                buttons.append(line)
        
        # Format the components
        formatted_instructions = "\n".join(instructions)
        formatted_buttons = "\n".join(buttons)
        
        return f"""In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value. Your answer should be a sequence of buttons in a boxed{{}} tag separated by '→', for example: boxed{{A → B → C}}.

{start_state}
{target_value}
Buttons:
{formatted_buttons}

Start your final output with your answer in a boxed{{}} tag and then give a short explanation of your reasoning. One sentence is enough."""

# For backward compatibility, also provide a function-based prompt
def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    """
    Function-based prompt for backward compatibility.
    """
    prompt_instance = QuantumLockPrompt()
    return prompt_instance(question, examples, metadata) 