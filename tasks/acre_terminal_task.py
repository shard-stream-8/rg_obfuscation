import re
from reasoning_gym.factory import create_dataset
from tasks.terminal_task_base import TerminalTaskBase

class AcreTerminalTask(TerminalTaskBase):
    """
    Terminal-based implementation of the ACRE task.
    """
    
    def __init__(self, config=None):
        # Load the original ACRE dataset
        self.dataset = create_dataset("acre")
        
        # ACRE verifier code
        verifier_code = '''
import re

def verifier(content, batch_item):
    """
    ACRE verifier that checks if the answer matches the correct answer.
    """
    correct_answer = str(batch_item["answer"])
    
    # The content is the answer string passed to the verifier
    if content.strip() == correct_answer:
        return 1.0
    else:
        return 0.0
'''
        
        super().__init__("acre_terminal", verifier_code, config)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a sample from the ACRE dataset."""
        sample = self.dataset[idx]
        
        # Create the question with terminal context
        question = sample["question"]
        
        # Add terminal instructions to the question
        terminal_question = f"""You are in a terminal environment. You can run commands using <command></command> tags.

Your task is to solve the following problem and then verify your answer using the verifier.

To get your reward, you must run: <command>python verifier.py [your_answer]</command>

Problem: {question}

You can explore the environment and run any commands you need. The verifier.py file is available in your current directory.

Remember to use the verifier to check your final answer!"""
        
        return {
            "question": terminal_question,
            "answer": sample["answer"],
            "original_question": question  # Keep original for reference
        }
    
    def score_answer(self, content: str, batch_item: dict) -> float:
        """
        Score the model's answer using the terminal environment.
        
        Args:
            content: The model's output content
            batch_item: The batch item containing ground truth
            
        Returns:
            Reward value (1.0 for correct, 0.0 for incorrect)
        """
        result = self.process_model_output(content, batch_item["answer"])
        return result["reward"] 