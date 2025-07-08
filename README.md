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
- **Trainer:** RL logic is in `reinforce/trainer.py` (supports KL penalty, batch size, etc.).
- **Logging:** Integrated with Weights & Biases via `wandb_logger.py`.

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

## Listing Available Verifiers and Prompts

- List verifiers: `python verifiers/list_verifiers.py`
- List prompts:   `python prompts/list_prompts.py`

## Adding New Tasks

- Add new tasks to the `tasks/` directory or extend via the `reasoning_gym` package.
- Use `tasks/list_tasks.py` to see available tasks.

---
For more details, see the docstrings in each module or browse the codebase. This repo is designed for extensibility and rapid experimentation with RL and reasoning tasks. 