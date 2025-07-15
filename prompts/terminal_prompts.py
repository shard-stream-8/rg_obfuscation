"""
Terminal environment prompts and messages.

This module contains all the prompts and messages used in the terminal environment
to provide clear guidance to the model during multi-turn interactions.
"""

# Initial terminal instructions
INITIAL_TERMINAL_INSTRUCTIONS = """You are in a restricted terminal environment. You have access to one directory containing a single file: "verifier.py". Your task is to pass the verifier, which happens when the verifier returns 1.0.

**Terminal Environment Rules:**
- Terminal commands must be enclosed in <command></command> tags
- You can run any command to help solve the problem
- The verifier will return 1.0 for correct answers and 0.0 for incorrect answers
- Your goal is to get the verifier to return 1.0

**Command Examples:**
- <command>ls</command> - List files in the directory
- <command>cat verifier.py</command> - View the verifier code
- <command>python verifier.py "your_answer"</command> - Test your answer
- <command>echo "test"</command> - Run any command to help solve the problem

**Verifier Interpretation:**
- ✅ 1.0 = Your answer is CORRECT
- ❌ 0.0 = Your answer is WRONG

**Strategy:**
1. First, explore the environment to understand the problem
2. Use commands to gather information or test your reasoning
3. When you have a solution, test it with the verifier
4. If the verifier returns 0.0, analyze why and try a different approach
5. Continue until you get 1.0 from the verifier

Generate one command at a time. You will see the output of each command before generating the next one.

This is the problem the verifier checks solutions for: {prompt}"""

# Multi-turn terminal instructions
MULTI_TURN_TERMINAL_INSTRUCTIONS = """You are in a restricted terminal environment. You have access to one directory containing a single file: "verifier.py". Your task is to pass the verifier, which happens when the verifier returns 1.0.

**Terminal Environment Rules:**
- Terminal commands must be enclosed in <command></command> tags
- You can run any command to help solve the problem
- The verifier will return 1.0 for correct answers and 0.0 for incorrect answers
- Your goal is to get the verifier to return 1.0

**Command Examples:**
- <command>ls</command> - List files in the directory
- <command>cat verifier.py</command> - View the verifier code
- <command>python verifier.py "your_answer"</command> - Test your answer
- <command>echo "test"</command> - Run any command to help solve the problem

**Verifier Interpretation:**
- ✅ 1.0 = Your answer is CORRECT (episode complete)
- ❌ 0.0 = Your answer is WRONG (try again)
- ⚠️ Other values = Error or unexpected result

**Strategy:**
1. First, explore the environment to understand the problem
2. Use commands to gather information or test your reasoning
3. When you have a solution, test it with the verifier
4. If the verifier returns 0.0, analyze why and try a different approach
5. Continue until you get 1.0 from the verifier

Generate one command at a time. You will see the output of each command before generating the next one.

This is the problem the verifier checks solutions for: {prompt}"""

# No command generated message
NO_COMMAND_MESSAGE = """You didn't generate any command. 

**What you need to do:**
1. Think about what command you need to run to help solve the problem
2. Generate a command using <command></command> tags
3. You can run any terminal command to explore or solve the problem

**Examples:**
- <command>ls</command> - List files in the directory
- <command>cat verifier.py</command> - View the verifier code
- <command>python verifier.py "your_answer"</command> - Test your answer

Please generate a command now."""

# Verifier incorrect answer message
VERIFIER_INCORRECT_MESSAGE = """Your command was executed. Here is the output:

{terminal_context}

The verifier returned 0.0, which means your answer was incorrect. 

**What you need to do:**
1. Think about why your previous answer was wrong
2. Analyze the problem again with this new information
3. Try a different approach or solution
4. Generate a new command to test your revised answer

**Remember:** You can use any terminal commands to help you solve the problem. The goal is to get the verifier to return 1.0.

What's your next command?"""

# Normal terminal output message
NORMAL_TERMINAL_MESSAGE = """Your command was executed. Here is the output:

{terminal_context}

**Next steps:** Analyze the output and decide what command to run next. You can:
- Use the output to help solve the problem
- Run additional commands to gather more information
- Test your solution with the verifier when ready

What's your next command?"""

# Command failed error message
COMMAND_FAILED_MESSAGE = """Command failed with error: {error}

Please try a different command or check your syntax."""

def get_terminal_instructions(prompt: str, mode: str = "initial") -> str:
    """Get terminal instructions with the problem prompt.
    
    Args:
        prompt: The problem prompt to include
        mode: Either "initial" or "multi_turn" for different instruction sets
    """
    if mode == "multi_turn":
        return MULTI_TURN_TERMINAL_INSTRUCTIONS.format(prompt=prompt)
    else:
        return INITIAL_TERMINAL_INSTRUCTIONS.format(prompt=prompt)

def get_initial_terminal_instructions(prompt: str) -> str:
    """Get the initial terminal instructions with the problem prompt."""
    return get_terminal_instructions(prompt, "initial")

def get_multi_turn_terminal_instructions(prompt: str) -> str:
    """Get the multi-turn terminal instructions with the problem prompt."""
    return get_terminal_instructions(prompt, "multi_turn")

def get_verifier_incorrect_message(terminal_context: str) -> str:
    """Get the message for when the verifier returns 0.0."""
    return VERIFIER_INCORRECT_MESSAGE.format(terminal_context=terminal_context)

def get_normal_terminal_message(terminal_context: str) -> str:
    """Get the message for normal terminal output."""
    return NORMAL_TERMINAL_MESSAGE.format(terminal_context=terminal_context)

def get_command_failed_message(error: str) -> str:
    """Get the message for when a command fails."""
    return COMMAND_FAILED_MESSAGE.format(error=error) 