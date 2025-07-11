# Reasoning Gym RL Infrastructure

## Repository Structure

```
rg_obfuscation/
├── config.yaml                # Main configuration file
├── models/                    # Model wrappers and interfaces
├── reinforce/                 # RL training logic (trainer, logger, utils, etc.)
├── tasks/                     # Task loading, templates, and utilities
├── train.py                   # Entry point for training
├── verifiers/                 # Custom verifiers for answer checking
├── prompts/                   # Custom prompt formatters (see below)
├── wandb_logger.py            # Weights & Biases logging
├── requirements.txt           # Python dependencies
```

## Training Infrastructure

- **Entry point:** `train.py` — launches RL training using the configuration in `config.yaml`.
- **Config:** Set model, task, verifier, prompt, and RL hyperparameters in `config.yaml`.
- **Trainer:** RL logic is in `reinforce/trainer.py` (supports KL penalty, batch size, gradient accumulation, etc.).
- **Logging:** Integrated with Weights & Biases via `wandb_logger.py`.

### Gradient Accumulation

The training system supports gradient accumulation to effectively increase the batch size without increasing memory usage:

- **Configuration:** Set `gradient_accumulation_steps` in `config.yaml` (default: 1)
- **Effective batch size:** `batch_size × gradient_accumulation_steps`
- **Memory efficient:** Gradients are accumulated over multiple episodes before updating model parameters
- **Learning rate:** Automatically scaled to maintain the same effective learning rate

Example configuration:
```yaml
batch_size: 16
gradient_accumulation_steps: 4  # Effective batch size = 64
learning_rate: 1e-5
```

## Custom Verifiers

Verifiers check if a model's output is correct for a given task.

- Place custom verifiers in `verifiers/` as `{task_name}_verifier.py`.
- Each verifier should provide a `verifier(content, batch_item)` function or a class inheriting from `BaseVerifier`.
- Verifiers are auto-registered and can be selected via `custom_verifier_path: "registry"` in `config.yaml`.
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
- Prompts are auto-registered and can be selected via `custom_prompt_path: "registry"` in `config.yaml`.
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

See [TERMINAL_MODE_README.md](./TERMINAL_MODE_README.md) for full details, migration notes, and advanced usage.

### Migration Note
If you previously used separate `_terminal` task or prompt files, you can now delete them and control terminal/multi-turn behavior via config flags. Just ensure your prompt file supports both modes (see `BasePrompt` and examples in `prompts/`).

## Listing Available Verifiers and Prompts

- List verifiers: `python verifiers/list_verifiers.py`
- List prompts:   `python prompts/list_prompts.py`

## Adding New Tasks

- Add new tasks to the `tasks/` directory or extend via the `reasoning_gym` package.
- Use `tasks/list_tasks.py` to see available tasks.

---
For more details, see the docstrings in each module or browse the codebase. This repo is designed for extensibility and rapid experimentation with RL and reasoning tasks. 