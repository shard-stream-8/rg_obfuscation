# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    from base_prompt import BasePrompt

class RansomNotePrompt(BasePrompt):
    """
    Custom prompt for the ransom_note task.
    """

    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """Generate prompt for the ransom_note task."""
        examples_text = ""
        if examples:
            examples_text = "Examples:\n"
            for i, ex in enumerate(examples, 1):
                if isinstance(ex, dict):
                    magazine = ex.get('magazine', '')
                    note = ex.get('note', '')
                    answer = ex.get('output', ex.get('answer', ''))
                    examples_text += f"Example {i}:\nMagazine: {magazine}\nNote: {note}\nAnswer: boxed{{{answer}}}\n\n"
                else:
                    examples_text += f"Example {i}: {ex}\n"
            examples_text += "\n"

        return f"""
You are solving the classic Ransom Note problem. 
You are given two strings:
1. magazine 
letters available\n2. ransom note – message to form\n\nEach letter in the magazine can be used at most once. Determine if the ransom note can be constructed from the magazine."""


Given two strings representing a ransom note and a magazine, return True if you can construct the ransom note using the letters in the magazine, and False otherwise.

Each letter in the magazine string can only be used once in your ransom note.

Ransom note: {ransom_note}
Magazine: {magazine}
# Function wrapper for backward compatibility
def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    return RansomNotePrompt()(question, examples, metadata) 