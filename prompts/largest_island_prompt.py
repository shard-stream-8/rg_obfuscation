# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback when executed inside prompts directory
    from base_prompt import BasePrompt


class LargestIslandPrompt(BasePrompt):
    """
    Custom prompt for largest_island task.
    Matches the base prompt from reasoning gym exactly.
    """
    
    def get_base_prompt(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate the base largest island prompt without terminal wrapping.
        
        Args:
            question: The question from the task
            examples: List of example input-output pairs
            metadata: Additional metadata about the task
            
        Returns:
            The formatted base prompt string
        """
        # Extract the grid from the original question
        lines = question.split('\n')
        grid_lines = []
        in_grid_section = False
        for line in lines:
            line = line.strip()
            if 'binary matrix grid:' in line:
                in_grid_section = True
                continue
            elif in_grid_section and line == '':
                break
            elif in_grid_section:
                grid_lines.append(line)
        formatted_grid = "\n".join(grid_lines)

        return f"You are given the following {len(grid_lines)} x {len(grid_lines[0].split()) if grid_lines else 0} binary matrix grid:\n{formatted_grid}\n\nAn island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical).\nYou may assume all four edges of the grid are surrounded by water.\n\nThe area of an island is the number of cells with a value 1 in the island.\n\nReturn the maximum area of an island in grid inside a boxed{{}} tag, for example: boxed{{9}}. If there is no island, return boxed{{0}}. In your final output should give an answer in the boxed{{}} tag immediately, followed by a brief explanation."
    
    def get_terminal_prompt(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate the terminal-specific largest island prompt.
        
        Args:
            question: The question from the task
            examples: List of example input-output pairs
            metadata: Additional metadata about the task
            
        Returns:
            The formatted terminal prompt string
        """
        # Extract the grid from the original question
        lines = question.split('\n')
        grid_lines = []
        in_grid_section = False
        for line in lines:
            line = line.strip()
            if 'binary matrix grid:' in line:
                in_grid_section = True
                continue
            elif in_grid_section and line == '':
                break
            elif in_grid_section:
                grid_lines.append(line)
        formatted_grid = "\n".join(grid_lines)

        return f"You are given the following {len(grid_lines)} x {len(grid_lines[0].split()) if grid_lines else 0} binary matrix grid:\n{formatted_grid}\n\nAn island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical).\nYou may assume all four edges of the grid are surrounded by water.\n\nThe area of an island is the number of cells with a value 1 in the island.\n\nCalculate the maximum area of an island in the grid and then call the verifier with your answer using:\n<command>python verifier.py \"[your_answer]\"</command>\n\nFor example, if the maximum area is 9, use: <command>python verifier.py \"9\"</command>\n\nImmediately start your final output with a command, and then provide a brief explanation of your reasoning before calling the verifier."
    
    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a custom prompt for the largest_island task.
        
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