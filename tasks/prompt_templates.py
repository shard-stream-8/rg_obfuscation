"""
Prompt template system for customizing prompts in reasoning gym tasks.
"""

import re
from typing import Dict, Any, Optional


class PromptTemplate:
    """
    A template system for customizing prompts in reasoning gym tasks.
    
    Supports the following placeholders:
    - {question}: The original question from the task
    - {task_name}: The name of the task
    - {task_description}: A description of the task
    - {answer_format}: Instructions for answer format
    """
    
    def __init__(self, template: str):
        """
        Initialize a prompt template.
        
        Args:
            template: The template string with placeholders
        """
        self.template = template
        self.placeholders = self._extract_placeholders(template)
    
    def _extract_placeholders(self, template: str) -> set:
        """Extract all placeholders from the template."""
        return set(re.findall(r'\{(\w+)\}', template))
    
    def format(self, **kwargs) -> str:
        """
        Format the template with the provided values.
        
        Args:
            **kwargs: Values to substitute for placeholders
            
        Returns:
            The formatted prompt
        """
        # Only use the placeholders that exist in the template
        template_kwargs = {k: v for k, v in kwargs.items() if k in self.placeholders}
        return self.template.format(**template_kwargs)


def get_task_description(task_name: str) -> str:
    """
    Get a description for a given task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Description of the task
    """
    descriptions = {
        "leg_counting": "counting the total number of legs for a list of animals",
        "arithmetic": "solving arithmetic problems",
        "logic": "solving logical reasoning problems",
        "geometry": "solving geometry problems",
        "sudoku": "solving Sudoku puzzles",
        "maze": "finding paths through mazes",
        "string_manipulation": "manipulating strings according to rules",
        "algorithm": "implementing algorithms",
        "game": "playing strategic games",
    }
    
    # Try exact match first
    if task_name in descriptions:
        return descriptions[task_name]
    
    # Try partial matches
    for key, desc in descriptions.items():
        if key in task_name.lower():
            return desc
    
    # Default description
    return f"solving {task_name} problems"


def get_answer_format_instructions(task_name: str) -> str:
    """
    Get answer format instructions for a given task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Instructions for answer format
    """
    formats = {
        "leg_counting": "Provide your answer in the format: boxed{answer}",
        "arithmetic": "Provide your answer as a number",
        "logic": "Provide your answer clearly",
        "geometry": "Provide your answer with appropriate units",
        "sudoku": "Provide your answer as a completed grid",
        "maze": "Provide your answer as a sequence of moves",
        "string_manipulation": "Provide your answer as the final string",
        "algorithm": "Provide your answer as the algorithm output",
        "game": "Provide your answer as the optimal move",
    }
    
    # Try exact match first
    if task_name in formats:
        return formats[task_name]
    
    # Try partial matches
    for key, format_inst in formats.items():
        if key in task_name.lower():
            return format_inst
    
    # Default format
    return "Provide your answer clearly"


def create_custom_prompt(
    original_question: str,
    task_name: str,
    template: Optional[str] = None
) -> str:
    """
    Create a custom prompt using a template.
    
    Args:
        original_question: The original question from the task
        task_name: Name of the task
        template: Optional custom template string
        
    Returns:
        The customized prompt
    """
    if template is None:
        return original_question
    
    # Process newlines in the template
    template = template.replace('\\n', '\n')
    
    # Create template object
    prompt_template = PromptTemplate(template)
    
    # Prepare context variables
    context = {
        "question": original_question,
        "task_name": task_name,
        "task_description": get_task_description(task_name),
        "answer_format": get_answer_format_instructions(task_name),
    }
    
    # Format the template
    return prompt_template.format(**context)


def get_default_templates() -> Dict[str, str]:
    """
    Get a dictionary of default templates for common tasks.
    
    Returns:
        Dictionary mapping task names to default templates
    """
    return {
        "leg_counting": (
            "You are an expert at counting animal legs. "
            "Your task is to count how many legs there are in total when given a list of animals.\n\n"
            "{question}\n\n"
            "Please think through this step by step and provide your answer in the format: boxed{{answer}}"
        ),
        "arithmetic": (
            "You are an expert mathematician. "
            "Please solve the following arithmetic problem step by step:\n\n"
            "{question}\n\n"
            "Show your work and provide the final answer."
        ),
        "logic": (
            "You are an expert at logical reasoning. "
            "Please solve the following logic problem step by step:\n\n"
            "{question}\n\n"
            "Explain your reasoning and provide your answer."
        ),
        "sudoku": (
            "You are an expert at solving Sudoku puzzles. "
            "Please solve the following Sudoku puzzle step by step:\n\n"
            "{question}\n\n"
            "Show your work and provide the completed grid."
        ),
        "general": (
            "You are an expert at {task_description}. "
            "Please solve the following problem step by step:\n\n"
            "{question}\n\n"
            "{answer_format}"
        ),
    }


def suggest_template_for_task(task_name: str) -> str:
    """
    Suggest a template for a given task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Suggested template string
    """
    default_templates = get_default_templates()
    
    # Try exact match first
    if task_name in default_templates:
        return default_templates[task_name]
    
    # Try partial matches
    for key, template in default_templates.items():
        if key in task_name.lower():
            return template
    
    # Return general template
    return default_templates["general"] 