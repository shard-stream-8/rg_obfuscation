#!/usr/bin/env python3
"""
Utility script to list all available tasks in reasoning_gym.
"""

from reasoning_gym.factory import DATASETS

def list_all_tasks():
    """List all available tasks grouped by category."""
    tasks = list(DATASETS.keys())
    tasks.sort()
    
    print(f"Total available tasks: {len(tasks)}")
    print("=" * 50)
    
    # Group tasks by category (based on naming patterns)
    categories = {
        'Arithmetic': [t for t in tasks if 'arithmetic' in t.lower() or 'sum' in t.lower() or 'counting' in t.lower()],
        'Logic': [t for t in tasks if 'logic' in t.lower() or 'puzzle' in t.lower() or 'sudoku' in t.lower()],
        'Geometry': [t for t in tasks if 'geometry' in t.lower() or 'matrix' in t.lower()],
        'Games': [t for t in tasks if 'game' in t.lower() or 'maze' in t.lower() or 'cube' in t.lower()],
        'Strings': [t for t in tasks if 'string' in t.lower() or 'word' in t.lower() or 'letter' in t.lower()],
        'Algorithms': [t for t in tasks if 'algorithm' in t.lower() or 'path' in t.lower() or 'sort' in t.lower()],
        'Other': [t for t in tasks if not any(keyword in t.lower() for keyword in ['arithmetic', 'logic', 'geometry', 'game', 'string', 'algorithm', 'sum', 'counting', 'puzzle', 'sudoku', 'matrix', 'maze', 'cube', 'word', 'letter', 'path', 'sort'])]
    }
    
    for category, task_list in categories.items():
        if task_list:
            print(f"\n{category} ({len(task_list)} tasks):")
            for task in task_list:
                print(f"  - {task}")
    
    print(f"\nTo use any task, simply set 'task_name: \"task_name\"' in your config.yaml")

def test_task_loading(task_name):
    """Test loading a specific task."""
    try:
        from tasks.task_loader import load_task
        dataset = load_task(task_name)
        print(f"✓ Successfully loaded '{task_name}' - {len(dataset)} samples")
        return True
    except Exception as e:
        print(f"✗ Failed to load '{task_name}': {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test loading a specific task
        task_name = sys.argv[1]
        test_task_loading(task_name)
    else:
        # List all tasks
        list_all_tasks() 