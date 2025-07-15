# Reasoning Gym RL Infrastructure

## Repository Structure

```
rg_obfuscation/
├── configs/                   # Configuration files with inheritance
│   ├── base.yaml             # Base configuration (inherited by others)
│   ├── debug.yaml            # Debug configuration
│   ├── island.yaml           # Island task configuration
│   ├── acre.yaml             # Acre task configuration
│   └── family.yaml           # Family relationships configuration
├── models/                    # Model wrappers and interfaces
├── reinforce/                 # RL training logic (trainer, logger, utils, etc.)
├── tasks/                     # Task loading, templates, and utilities
├── train.py                   # Entry point for training
├── prompts/                   # Custom prompt formatters (see below)
├── wandb_logger.py            # Weights & Biases logging
├── terminal_env.py            # Terminal environment for interactive tasks
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Training Infrastructure

- **Entry point:** `train.py` — launches RL training using the configuration in `config.yaml`.
- **Config:** Set model, task, verifier, prompt, and RL hyperparameters in config files.
- **Trainer:** RL logic is in `reinforce/trainer.py` (supports KL penalty, batch size, gradient accumulation, etc.).
- **Logging:** Integrated with Weights & Biases via `wandb_logger.py` and improved rollout logging.
- **Batched Multi-Turn Training:** Process multiple episodes in parallel for each turn, improving training efficiency.

### Configuration System

The system now uses a **config inheritance** system for better maintainability:

- **Base Config:** `configs/base.yaml` contains all common settings
- **Task Configs:** Individual task configs only override necessary values
- **Inheritance:** All configs automatically inherit from `base.yaml`

Example task config (`configs/acre.yaml`):
```yaml
# Acre configuration - inherits from base.yaml
# Override only the values that differ from base config

# task configuration
task_name: "acre"
```

### Gradient Accumulation

The training system supports gradient accumulation to effectively increase the batch size without increasing memory usage:

- **Configuration:** Set `gradient_accumulation_steps` in config (default: 1)
- **Effective batch size:** `batch_size × gradient_accumulation_steps`
- **Memory efficient:** Gradients are accumulated over multiple episodes before updating model parameters
- **Learning rate:** Automatically scaled to maintain the same effective learning rate

Example configuration:
```yaml
batch_size: 16
gradient_accumulation_steps: 4  # Effective batch size = 64
learning_rate: 1e-5
```

### Batched Multi-Turn Training

The multi-turn training system now supports batching to process multiple episodes in parallel for each turn:

- **Parallel Generation:** Generate turn n for all episodes in a batch simultaneously
- **Early Completion Handling:** Episodes that complete early (success or turn limit) are removed from the active batch
- **Memory Efficient:** Only active episodes continue to the next turn
- **Training Efficiency:** Reduces the number of forward passes needed for multi-turn episodes

**How it works:**
1. Initialize `batch_size` episodes with different prompts
2. For each turn, generate responses for all active episodes in parallel
3. Process terminal commands and update episode states
4. Remove completed episodes from the active batch
5. Continue until all episodes are complete or turn limit reached

**Benefits:**
- **Faster Training:** Parallel generation reduces wall-clock time
- **Better GPU Utilization:** Batched forward passes are more efficient
- **Consistent Episodes:** All episodes in a batch progress through turns together
- **Scalable:** Works with any batch size configuration

Example configuration for batched multi-turn training:
```yaml
task_name: "acre"
use_terminal: true
enable_multi_turn: true
max_turns: 5
batch_size: 8  # Process 8 episodes in parallel
gradient_accumulation_steps: 2  # Effective batch size = 16
```

### Improved Rollout Logging

The system now provides **multiple logging formats** for better analysis:

- **JSON Format:** Standard JSON for programmatic analysis
- **Readable JSON:** Enhanced JSON with turn numbers and better formatting
- **Super Readable Text:** Human-readable text format with preserved formatting and newlines

All formats focus on essential information:
- Conversation dialogue (main focus)
- Rewards and metrics
- Multi-turn metrics (when applicable)

## Custom Verifiers

Verifiers check if a model's output is correct for a given task.

- Place custom verifiers in `verifiers/` as `{task_name}_verifier.py`.
- Each verifier should provide a `verifier(content, batch_item)` function or a class inheriting from `BaseVerifier`.
- Verifiers are auto-registered and can be selected via `custom_verifier_path: "registry"` in config.
- Example:
  ```python
  def verifier(content, batch_item):
      # Return 1.0 if correct, 0.0 if not
      return float(batch_item["answer"] in content)
  ```

## Custom Prompts

Prompts control how questions and examples are presented to the model.

- Place custom prompts in `prompts/` as `{task_name}_prompt.py`.
- Each prompt should provide a `prompt(question, examples, metadata)` function or a class inheriting from `BasePrompt`.
- Prompts are auto-registered and can be selected via `custom_prompt_path: "registry"` in config.
- Example:
  ```python
  def prompt(question, examples, metadata=None):
      formatted_examples = "\n".join(f"{ex['input']} → {ex['output']}" for ex in examples)
      return f"Examples:\n{formatted_examples}\n\nQuestion: {question}"
  ```

## Terminal and Multi-Turn Mode

You can now enable terminal-based and multi-turn task environments for any Reasoning Gym task using configuration flags—no need for separate `_terminal` task files or prompt files.

### Configuration Flags

- `use_terminal`: Enable terminal mode (default: `false`).
- `enable_multi_turn`: Enable multi-turn interaction (default: `false`).
- `max_turns`: Maximum number of turns for multi-turn mode (default: `10`).

### Example Configurations

**Regular (non-terminal) mode:**
```yaml
task_name: "acre"
use_terminal: false
enable_multi_turn: false
```

**Single-turn terminal mode:**
```yaml
task_name: "acre"
use_terminal: true
enable_multi_turn: false
```

**Multi-turn terminal mode:**
```yaml
task_name: "acre"
use_terminal: true
enable_multi_turn: true
max_turns: 3
```

### How it Works
- The system wraps any task in a terminal environment if `use_terminal` is true.
- Multi-turn and single-turn are handled by the same wrapper.
- Prompts can implement both terminal and non-terminal logic in a single file (see `BasePrompt`).
- No need for `[task]_terminal_task.py` or `[task]_terminal_prompt.py` files.
- Backward compatibility: legacy `_terminal` task names still work.

### Terminal Environment Features

- **Isolated Environment:** Each episode gets a fresh terminal environment
- **Command Extraction:** Automatically extracts commands from `<command></command>` tags
- **Verifier Integration:** Built-in verifier system for answer validation
- **Context Preservation:** Maintains conversation history across turns
- **Error Handling:** Robust error handling for command execution
- **Batched Multi-Turn Training:** Process multiple episodes in parallel for each turn

## Code Quality Improvements

Recent refactoring has significantly improved code quality:

### Consolidated Functions
- **Rollout Logging:** Unified 3 separate functions into 1 with format parameter
- **KL Penalty:** Centralized KL penalty calculation in `reinforce/kl_penalty.py`
- **Model Loading:** Eliminated duplicate model loading logic
- **Terminal Prompts:** Consolidated redundant terminal instruction functions

### Configuration Inheritance
- **Base Config:** `configs/base.yaml` contains all common settings
- **Minimal Overrides:** Task configs only specify what differs from base
- **Easy Maintenance:** Update common settings in one place

### Improved Logging
- **Multiple Formats:** JSON, readable JSON, and super-readable text
- **Preserved Formatting:** Newlines and formatting maintained in logs
- **Essential Focus:** Only logs conversation dialogue, rewards, and metrics

## Listing Available Verifiers and Prompts

- List verifiers: `python verifiers/list_verifiers.py`
- List prompts:   `python prompts/list_prompts.py`

## Adding New Tasks

- Add new tasks to the `tasks/` directory or extend via the `reasoning_gym` package.
- Use `tasks/list_tasks.py` to see available tasks.

## Quick Start

1. **Choose a config:** Select from `configs/` or create your own
2. **Run training:** `python train.py configs/acre.yaml`
3. **Monitor logs:** Check `rollouts/` directory for detailed episode logs
4. **View metrics:** Weights & Biases integration for experiment tracking

## Migration Notes

- **Config Files:** Old single config files can be converted to use inheritance
- **Terminal Tasks:** Legacy `_terminal` task names still work but are no longer needed
- **Logging:** Rollout logs now have multiple formats for better analysis

---
For more details, see the docstrings in each module or browse the codebase. This repo is designed for extensibility and rapid experimentation with RL and reasoning tasks. 