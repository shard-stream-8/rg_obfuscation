#!/usr/bin/env python3
"""
Utility script to list all available custom verifiers.
"""

# Import registry using absolute import
try:
    from verifiers.registry import registry
except ImportError:
    # Fallback for when running from within the verifiers directory
    from registry import registry

def main():
    """List all available verifiers."""
    print("Available custom verifiers:")
    print("=" * 40)
    
    verifiers = registry.list_available_verifiers()
    
    if not verifiers:
        print("No custom verifiers found.")
        print("\nTo add a verifier:")
        print("1. Create a file named '{task_name}_verifier.py' in the verifiers directory")
        print("2. Define a 'verifier' function or a class that inherits from BaseVerifier")
        print("3. The verifier will be automatically registered")
    else:
        for task_name in sorted(verifiers):
            print(f"- {task_name}")
        
        print(f"\nTotal: {len(verifiers)} verifier(s)")
        print("\nTo use a verifier, set custom_verifier_path='registry' in config.yaml")

if __name__ == "__main__":
    main() 