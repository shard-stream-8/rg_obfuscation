import re
import ast
from typing import Union, List

def verifier(content: str, batch_item: dict) -> float:
    """
    Verify if the model's output contains the correct answer for list_functions task.
    
    This verifier:
    1. Removes thinking content (anything between <think> and </think> tags)
    2. Checks if the correct answer appears anywhere in the remaining output
    3. Handles various list formats and whitespace variations
    4. Doesn't penalize additional tokens or formatting
    
    Args:
        content: The model's output content
        batch_item: Dictionary containing the question and correct answer
        
    Returns:
        float: 1.0 if correct answer is found, 0.0 otherwise
    """
    correct_answer = batch_item["answer"]
    
    # Remove thinking content
    content_without_thinking = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Clean up the content
    content_clean = content_without_thinking.strip()
    
    # Try to parse the correct answer as a Python literal
    try:
        expected_list = ast.literal_eval(correct_answer)
    except (ValueError, SyntaxError):
        # If parsing fails, treat as string
        expected_list = correct_answer
    
    # If expected_list is a list, convert it to string representation for comparison
    if isinstance(expected_list, list):
        expected_str = str(expected_list)
    else:
        expected_str = str(expected_list)
    
    # Check if the expected answer appears in the content
    # Use case-insensitive matching and handle whitespace variations
    expected_normalized = re.escape(expected_str)
    
    # Look for the answer in the content
    if re.search(expected_normalized, content_clean, re.IGNORECASE):
        return 1.0
    
    # Also try with different list formats
    # Remove spaces around brackets and commas
    expected_compact = expected_str.replace(' ', '').replace(',]', ']').replace('[,', '[')
    content_compact = content_clean.replace(' ', '').replace(',]', ']').replace('[,', '[')
    
    if expected_compact in content_compact:
        return 1.0
    
    # Try with different bracket styles
    expected_alt = expected_str.replace('[', '(').replace(']', ')')
    if expected_alt in content_clean:
        return 1.0
    
    # Try with curly braces
    expected_curly = expected_str.replace('[', '{').replace(']', '}')
    if expected_curly in content_clean:
        return 1.0
    
    return 0.0


class ListFunctionsVerifier:
    """
    Class-based verifier for list_functions task.
    """
    
    def __call__(self, content: str, batch_item: dict) -> float:
        """
        Call the verifier function.
        
        Args:
            content: The model's output content
            batch_item: Dictionary containing the question and correct answer
            
        Returns:
            float: 1.0 if correct answer is found, 0.0 otherwise
        """
        return verifier(content, batch_item)
    
    def verify(self, content: str, batch_item: dict) -> float:
        """
        Alternative method name for compatibility.
        
        Args:
            content: The model's output content
            batch_item: Dictionary containing the question and correct answer
            
        Returns:
            float: 1.0 if correct answer is found, 0.0 otherwise
        """
        return verifier(content, batch_item) 