# Import BasePrompt
from prompts.base_prompt import BasePrompt

class AcrePrompt(BasePrompt):
    """
    Custom prompt for acre task.
    
    Formats the examples with command-based answers and uses the new prompt instructions.
    """
    
    def __init__(self, use_terminal=False, enable_multi_turn=False):
        """
        Initialize the ACRE prompt.
        
        Args:
            use_terminal: Whether to use terminal mode
            enable_multi_turn: Whether to enable multi-turn mode
        """
        super().__init__(use_terminal, enable_multi_turn)
    
    def get_base_prompt(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate the base ACRE prompt without terminal wrapping.
        
        Args:
            question: The original question from the task
            examples: List of example input-output pairs (not used for ACRE as examples are in question)
            metadata: Additional metadata about the task (optional)
            
        Returns:
            The formatted base prompt string
        """
        # Extract the original question from the terminal-wrapped version
        if "This is the problem the verifier checks solutions for:" in question:
            # Extract the original question after this line
            lines = question.split('\n')
            for i, line in enumerate(lines):
                if "This is the problem the verifier checks solutions for:" in line:
                    original_question = '\n'.join(lines[i+1:]).strip()
                    break
            else:
                original_question = question
        else:
            original_question = question
                
        # For ACRE task, the examples are already included in the question
        # We need to extract them and format them properly
        
        # Split the question into lines to parse it
        lines = original_question.split('\n')
        
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
        
        if self.use_terminal:
            # Terminal version with command tags
            return f"""Certain objects (called 'blickets') have the hidden property of activating a detector, causing its light to turn on.

Each example shows the results of placing different combinations of objects on the detector. Each object is described by color, material and shape. Your task is to determine whether a new combination of objects will cause the detector to activate.

After observing the previous examples, respond with:
- <command>python verifier.py "on"</command> if you can determine the detector light will turn on
- <command>python verifier.py "off"</command> if you can determine the detector light will stay off  
- <command>python verifier.py "undetermined"</command> if there is insufficient evidence to reach a conclusion

Previous experimental results:
{formatted_examples}

New test case:
{new_test_case}

Answer immediately with the appropriate command and then provide a brief explanation of your answer.

What is the detector light status?"""
        else:
            # Non-terminal version
            return f"""Certain objects (called 'blickets') have the hidden property of activating a detector, causing its light to turn on.

Each example shows the results of placing different combinations of objects on the detector. Each object is described by color, material and shape. Your task is to determine whether a new combination of objects will cause the detector to activate.

After observing the previous examples, respond with:
- "on" if you can determine the detector light will turn on
- "off" if you can determine the detector light will stay off  
- "undetermined" if there is insufficient evidence to reach a conclusion

Previous experimental results:
{formatted_examples}

New test case:
{new_test_case}

Answer immediately with the appropriate response and then provide a brief explanation of your answer.

What is the detector light status?"""
    
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
        if self.use_terminal:
            return self.get_terminal_prompt(question, examples, metadata)
        else:
            return self.get_base_prompt(question, examples, metadata)

# For backward compatibility, also provide a function-based prompt
def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    """
    Function-based prompt for backward compatibility.
    """
    # Check if we should use terminal mode based on metadata
    use_terminal = metadata.get('use_terminal', False) if metadata else False
    enable_multi_turn = metadata.get('enable_multi_turn', False) if metadata else False
    
    print(f"DEBUG: prompt() called with use_terminal={use_terminal}, enable_multi_turn={enable_multi_turn}")
    print(f"DEBUG: metadata keys: {list(metadata.keys()) if metadata else 'None'}")
    
    prompt_instance = AcrePrompt(use_terminal=use_terminal, enable_multi_turn=enable_multi_turn)
    result = prompt_instance(question, examples, metadata)
    
    print(f"DEBUG: prompt() returning result with length {len(result)}")
    print(f"DEBUG: result contains '<command>' = {'<command>' in result}")
    
    return result 