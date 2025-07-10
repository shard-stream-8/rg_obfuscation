# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback for when running from within the prompts directory
    from base_prompt import BasePrompt

class CourseSchedulePrompt(BasePrompt):
    """
    Custom prompt for course_schedule task.
    
    Initially matches the reasoning gym default prompt exactly.
    """
    
    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a custom prompt for the course_schedule task.
        
        Args:
            question: The original question from the task
            examples: List of example input-output pairs (optional)
            metadata: Additional metadata about the task (optional)
            
        Returns:
            The formatted prompt string
        """
        # For course_schedule task, the question already contains the complete prompt
        # from reasoning gym's QUESTION_TEMPLATE. We return it as-is initially.
        return question

# For backward compatibility, also provide a function-based prompt
def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    """
    Function-based prompt for backward compatibility.
    """
    prompt_instance = CourseSchedulePrompt()
    return prompt_instance(question, examples, metadata) 