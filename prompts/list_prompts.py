#!/usr/bin/env python3
"""
Utility script to list all available custom prompts.
"""

# Import registry using absolute import
try:
    from prompts.registry import registry
except ImportError:
    # Fallback for when running from within the prompts directory
    from registry import registry

def main():
    """List all available prompts."""
    print("Available custom prompts:")
    print("=" * 40)
    
    prompts = registry.list_available_prompts()
    
    if not prompts:
        print("No custom prompts found.")
        print("\nTo add a prompt:")
        print("1. Create a file named '{task_name}_prompt.py' in the prompts directory")
        print("2. Define a 'prompt' function or a class that inherits from BasePrompt")
        print("3. The prompt will be automatically registered")
    else:
        for task_name in sorted(prompts):
            print(f"- {task_name}")
        
        print(f"\nTotal: {len(prompts)} prompt(s)")
        print("\nTo use a prompt, set custom_prompt_path='registry' in config.yaml")

if __name__ == "__main__":
    main() 