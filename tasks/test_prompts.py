#!/usr/bin/env python3
"""
Utility script to test and explore prompt templates for reasoning gym tasks.
"""

import sys
import yaml
from prompt_templates import (
    create_custom_prompt, 
    suggest_template_for_task, 
    get_default_templates,
    get_task_description,
    get_answer_format_instructions
)
from task_loader import load_task


def test_prompt_template(task_name: str, template: str = None, num_examples: int = 3):
    """
    Test a prompt template with actual task examples.
    
    Args:
        task_name: Name of the task to test
        template: Optional custom template to test
        num_examples: Number of examples to show
    """
    try:
        # Load the task
        task = load_task(task_name)
        
        # Get suggested template if none provided
        if template is None:
            template = suggest_template_for_task(task_name)
            print(f"Using suggested template for '{task_name}':")
        else:
            print(f"Testing custom template for '{task_name}':")
        
        print(f"Template: {template}")
        print("=" * 80)
        
        # Show examples
        for i in range(min(num_examples, len(task))):
            example = task[i]
            original_question = example["question"]
            correct_answer = example["answer"]
            
            # Create custom prompt
            custom_prompt = create_custom_prompt(
                original_question=original_question,
                task_name=task_name,
                template=template
            )
            
            print(f"\nExample {i+1}:")
            print(f"Original question: {original_question}")
            print(f"Correct answer: {correct_answer}")
            print(f"Custom prompt:")
            print("-" * 40)
            print(custom_prompt)
            print("-" * 40)
            
    except Exception as e:
        print(f"Error testing template: {e}")


def show_available_templates():
    """Show all available default templates."""
    templates = get_default_templates()
    
    print("Available default templates:")
    print("=" * 50)
    
    for task_name, template in templates.items():
        print(f"\n{task_name.upper()}:")
        print("-" * 30)
        print(template)
        print()


def show_task_info(task_name: str):
    """Show information about a specific task."""
    try:
        task = load_task(task_name)
        description = get_task_description(task_name)
        answer_format = get_answer_format_instructions(task_name)
        
        print(f"Task: {task_name}")
        print(f"Description: {description}")
        print(f"Answer format: {answer_format}")
        print(f"Number of examples: {len(task)}")
        
        # Show a sample question
        if len(task) > 0:
            sample = task[0]
            print(f"\nSample question: {sample['question']}")
            print(f"Sample answer: {sample['answer']}")
            
    except Exception as e:
        print(f"Error getting task info: {e}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_prompts.py <task_name> [template] [num_examples]")
        print("  python test_prompts.py --templates")
        print("  python test_prompts.py --info <task_name>")
        print("\nExamples:")
        print("  python test_prompts.py leg_counting")
        print("  python test_prompts.py leg_counting 'You are an expert at {task_name}. {question}' 2")
        print("  python test_prompts.py --templates")
        print("  python test_prompts.py --info arithmetic")
        return
    
    if sys.argv[1] == "--templates":
        show_available_templates()
        return
    
    if sys.argv[1] == "--info":
        if len(sys.argv) < 3:
            print("Please provide a task name for --info")
            return
        show_task_info(sys.argv[2])
        return
    
    task_name = sys.argv[1]
    template = sys.argv[2] if len(sys.argv) > 2 else None
    num_examples = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    test_prompt_template(task_name, template, num_examples)


if __name__ == "__main__":
    main() 