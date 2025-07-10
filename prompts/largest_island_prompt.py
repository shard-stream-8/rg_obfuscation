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
    
    def __call__(self, question: str, examples: list = None, metadata: dict = None) -> str:
        """
        Generate a custom prompt for the largest_island task that matches the base prompt.
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


# For backward compatibility, also provide a function-based prompt
def prompt(question: str, examples: list = None, metadata: dict = None) -> str:
    """
    Function-based prompt for backward compatibility.
    """
    prompt_instance = LargestIslandPrompt()
    return prompt_instance(question, examples, metadata) 