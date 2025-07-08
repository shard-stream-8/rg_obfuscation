# Custom Verifiers

This directory contains custom verifiers for different tasks in the reinforcement learning system.

## Overview

Custom verifiers allow you to define task-specific verification logic that checks if the model's output is correct. The system supports both function-based and class-based verifiers.

## Usage

### Method 1: Using the Registry (Recommended)

1. Set `custom_verifier_path: "registry"` in your `config.yaml`
2. The system will automatically load the appropriate verifier for your task

```yaml
model_name: "Qwen/Qwen3-4B"
task_name: "leg_counting"
custom_verifier_path: "registry"  # Use registry system
# ... other config options
```

### Method 2: Direct File Path

1. Set `custom_verifier_path` to the path of your verifier file in `config.yaml`

```yaml
model_name: "Qwen/Qwen3-4B"
task_name: "leg_counting"
custom_verifier_path: "verifiers/leg_counting_verifier.py"
# ... other config options
```

## Creating Custom Verifiers

### Function-Based Verifier (Simple)

Create a file named `{task_name}_verifier.py` in the `verifiers` directory:

```python
def verifier(content, batch_item):
    """
    Verify if the model's output is correct.
    
    Args:
        content: The model's output content
        batch_item: The batch item containing the question and correct answer
        
    Returns:
        float: Score between 0.0 and 1.0 indicating correctness
    """
    correct_answer = str(batch_item["answer"])
    
    # Your verification logic here
    if correct_answer in content:
        return 1.0
    else:
        return 0.0
```

### Class-Based Verifier (Advanced)

For more complex verifiers, inherit from `BaseVerifier`:

```python
from .base_verifier import BaseVerifier

class MyTaskVerifier(BaseVerifier):
    def verify(self, content, batch_item):
        """
        Verify if the model's output is correct.
        
        Args:
            content: The model's output content
            batch_item: The batch item containing the question and correct answer
            
        Returns:
            float: Score between 0.0 and 1.0 indicating correctness
        """
        # Your verification logic here
        return 1.0 if self._check_answer(content, batch_item) else 0.0
    
    def _check_answer(self, content, batch_item):
        # Helper method for complex verification logic
        pass
```

## Available Verifiers

### leg_counting_verifier.py

Checks if the output contains the correct answer in the format `boxed{[answer]}`. Ignores chain of thought content (anything between `<think>` and `</think>` tags).

**Features:**
- Removes thinking content before verification
- Case-insensitive matching
- Exact answer matching with proper escaping

**Example:**
- Correct answer: "42"
- Valid output: "The answer is boxed{42}"
- Invalid output: "The answer is 42" (missing boxed format)

## Listing Available Verifiers

Run the utility script to see all available verifiers:

```bash
python verifiers/list_verifiers.py
```

## Verifier Interface

All verifiers must implement the following interface:

```python
def verify(content: str, batch_item: dict) -> float:
    """
    Args:
        content: The model's output content (string)
        batch_item: Dictionary containing:
            - "question": The input question
            - "answer": The correct answer
            
    Returns:
        float: Score between 0.0 and 1.0
            - 1.0: Correct answer
            - 0.0: Incorrect answer
    """
    pass
```

## Best Practices

1. **Ignore thinking content**: Use regex to remove `<think>...</think>` sections before verification
2. **Handle edge cases**: Consider whitespace, case sensitivity, and formatting variations
3. **Use proper escaping**: When using regex patterns, escape special characters in the answer
4. **Provide clear feedback**: Add print statements or logging for debugging
5. **Test thoroughly**: Create test cases with various output formats

## Troubleshooting

### Verifier not found
- Check that the file is named correctly: `{task_name}_verifier.py`
- Ensure the file is in the `verifiers` directory
- Verify that the `verifier` function or class is properly defined

### Import errors
- Make sure all required imports are available
- Check that the verifier doesn't have syntax errors
- Verify that the registry can load the verifier file

### Verification not working
- Add debug prints to see what content is being verified
- Check that the batch_item structure matches expectations
- Verify that the answer format in the output matches your verification logic 