# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback when executed inside prompts directory
    from base_prompt import BasePrompt


class FamilyRelationshipsPrompt(BasePrompt):
    """Custom prompt for the family_relationships task.

    Guides the model to think first, then output an explanation and the final
    answer in a strict format compatible with the verifier.
    """

    def get_base_prompt(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate the base family relationships prompt without terminal wrapping.
        
        Args:
            question: The question from the task
            examples: List of example input-output pairs
            metadata: Additional metadata about the task
            
        Returns:
            The formatted base prompt string
        """
        instructions = (
            "You are an expert at determining family relationships.\n"
            "Read the scenario carefully and reason step by step inside the <think> and </think> tags.\n"
            "When you have finished thinking, close the </think> tag and then provide your response in the exact "
            "following format (replace the placeholders in brackets):\n\n"
            "### Explanation\n"
            "[brief explanation of your reasoning]\n\n"
            "### Answer\n"
            "[the relationship in one word or hyphenated term]\n\n"
            "Do not output anything else outside this exact format."
        )

        # Return the complete prompt with the question and an open <think> tag
        return f"{instructions}\n\n{question}\n\n<think>"
    
    def get_terminal_prompt(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate the terminal-specific family relationships prompt.
        
        Args:
            question: The question from the task
            examples: List of example input-output pairs
            metadata: Additional metadata about the task
            
        Returns:
            The formatted terminal prompt string
        """
        instructions = (
            "You are an expert at determining family relationships.\n"
            "Read the scenario carefully and reason step by step inside the <think> and </think> tags.\n"
            "When you have finished thinking, close the </think> tag and then provide your response using the verifier command.\n\n"
            "Use the following format:\n\n"
            "### Explanation\n"
            "[brief explanation of your reasoning]\n\n"
            "### Answer\n"
            "<command>python verifier.py \"[the relationship in one word or hyphenated term]\"</command>\n\n"
            "Do not output anything else outside this exact format."
        )

        # Return the complete prompt with the question and an open <think> tag
        return f"{instructions}\n\n{question}\n\n<think>"

    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a custom prompt for the family relationships task.
        
        Args:
            question: The question from the task
            examples: List of example input-output pairs
            metadata: Additional metadata about the task
            
        Returns:
            The formatted prompt string
        """
        if self.use_terminal:
            return self.get_terminal_prompt(question, examples, metadata)
        else:
            return self.get_base_prompt(question, examples, metadata)