import random
import torch
import yaml

from models.qwen3 import load_qwen3_model, prepare_thinking_input
from reinforce.logit_processor import BatchThinkingTokenBudgetProcessor
from tasks.task_loader import load_task
from tasks.prompt_templates import create_custom_prompt
from reinforce.utils import zero_special_token_grads
from wandb_logger import WandbLogger
from reinforce.rollout_logger import RolloutLogger

class Config:
    def __init__(self, d):
        self.__dict__.update(d)
    def __getitem__(self, k):
        return getattr(self, k)
    def __iter__(self):
        return iter(self.__dict__)
    def items(self):
        return self.__dict__.items()
    def __contains__(self, k):
        return k in self.__dict__


def train(config_path: str = "config.yaml") -> None:
    with open(config_path, "r") as f:
        config = Config(yaml.safe_load(f))

    # Set all RNGs for reproducibility
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Policy network -------------------------------------------------------
    model, tokenizer, device = load_qwen3_model(config.model_name, config.device)

    # Reference network for KL penalty (frozen parameters) -----------------
    ref_model = None
    # Always load reference model for KL penalty calculation (even if coefficient is 0)
    ref_model, _, _ = load_qwen3_model(config.model_name, config.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # RL task & logging helpers -------------------------------------------
    task = load_task(config.task_name, config.custom_verifier_path)
    wandb_logger   = WandbLogger(config)
    rollout_logger = RolloutLogger(config)

    # Load custom prompt at the start --------------------------------------
    custom_prompt = None
    if hasattr(config, 'custom_prompt_path') and config.custom_prompt_path:
        if config.custom_prompt_path == "registry":
            # Use the registry system
            try:
                from prompts.registry import registry
                custom_prompt = registry.get_prompt(config.task_name)
                if custom_prompt is not None:
                    print(f"Loaded custom prompt for task '{config.task_name}' from registry")
                else:
                    print(f"No custom prompt found in registry for task '{config.task_name}', using default")
            except ImportError:
                print("Prompt registry not available, falling back to default prompt")
        else:
            # Use the old file-based approach
            import importlib.util
            import sys
            spec = importlib.util.spec_from_file_location("custom_prompt", config.custom_prompt_path)
            custom_prompt_module = importlib.util.module_from_spec(spec)
            sys.modules["custom_prompt"] = custom_prompt_module
            spec.loader.exec_module(custom_prompt_module)
            
            if hasattr(custom_prompt_module, 'prompt'):
                custom_prompt = custom_prompt_module.prompt
                print(f"Loaded custom prompt from {config.custom_prompt_path}")
            else:
                print(f"Warning: No 'prompt' function found in {config.custom_prompt_path}")

    # Optimiser ------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=1e-2,
    )

    # Processor that constrains the number of <think> tokens ---------------
    logit_processor = BatchThinkingTokenBudgetProcessor(
        tokenizer,
        max_thinking_tokens=config.max_thinking_tokens,
        batch_size=config.batch_size,
        min_thinking_tokens=config.min_thinking_tokens,
    )

    # Gradient accumulation variables
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    accumulated_loss = 0.0
    accumulated_rewards = []
    accumulated_advantages = []
    accumulated_kl_penalties = []
    accumulated_episodes = []

    for episode in range(config.num_episodes):
        model.eval()

        # Reset logit processor state for new episode
        logit_processor.reset()

        batch_indices = [random.randrange(len(task)) for _ in range(config.batch_size)]
        batch        = [task[idx] for idx in batch_indices]
        
        # Apply custom prompt if available
        if custom_prompt is not None:
            prompts = [
                custom_prompt(b["question"], examples=None, metadata={"task_name": config.task_name})
                if callable(custom_prompt) else custom_prompt(b["question"])
                for b in batch
            ]
        elif hasattr(config, 'prompt_template') and config.prompt_template:
            # Fallback to the old prompt_template system
            prompts = [
                create_custom_prompt(
                    original_question=b["question"],
                    task_name=config.task_name,
                    template=config.prompt_template
                ) for b in batch
            ]
        else:
            prompts = [b["question"] for b in batch]
            
        targets      = [b["answer"]   for b in batch]

        prompt_inputs = [prepare_thinking_input(tokenizer, p, enable_thinking=True) for p in prompts]
        model_inputs  = tokenizer(prompt_inputs, return_tensors="pt", padding=True).to(device)

        prompt_lens = (model_inputs.input_ids != tokenizer.pad_token_id).sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=config.max_new_tokens,
                logits_processor=[logit_processor],
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = outputs.sequences  # (B, prompt_len + gen_len)

        thinking_contents, contents = [], []
        end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        start_think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        
        for seq, p_len in zip(generated_ids, prompt_lens):
            response_ids = seq[p_len:].tolist()
            
            # Find <think> and </think> tag positions
            try:
                thinking_start = response_ids.index(start_think_token_id)
            except ValueError:
                thinking_start = None
            try:
                thinking_end = len(response_ids) - response_ids[::-1].index(end_think_token_id)
            except ValueError:
                thinking_end = None
            
            if thinking_start is not None:
                if thinking_end is not None and thinking_end > thinking_start:
                    # Both tags present and in order
                    thinking_ids = response_ids[thinking_start + 1:thinking_end - 1]  # exclude <think> and </think>
                    content_ids = response_ids[thinking_end:]
                    thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
                    answer = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
                else:
                    # Only <think> present, no </think>
                    thinking_ids = response_ids[thinking_start + 1:]
                    thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
                    answer = ""
            else:
                # No <think> tag, treat all as answer
                thinking = ""
                answer = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            # Clean up any prompt template fragments that might have leaked through
            prompt_fragments = [
                "Write in as much detail as is useful inside think tags, but give only a brief explanation in your final output.",
                "assistant",
                "user:",
                "assistant:"
            ]
            for fragment in prompt_fragments:
                thinking = thinking.replace(fragment, "").strip()
                answer = answer.replace(fragment, "").strip()
            thinking_contents.append(thinking)
            contents.append(answer)

        rewards = torch.tensor(
            [task.score_answer(c, b) for b, c in zip(batch, contents)],
            dtype=torch.float32,
            device=device,
        )

        model.train()
        
        # Only zero gradients on the first accumulation step
        if (episode % gradient_accumulation_steps) == 0:
            optimizer.zero_grad()

        # Build a tensor of prompt + generated tokens for every example ----
        full_ids_list, start_indices = [], []
        for prompt_ids, seq in zip(model_inputs.input_ids, generated_ids):
            p_len = prompt_ids.size(0)
            start_indices.append(p_len - 1)  # -1 because targets are shifted by 1
            full_ids_list.append(seq)        # seq already contains prompt+gen
        full_ids = torch.nn.utils.rnn.pad_sequence(
            full_ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        full_ids = full_ids.to(device)

        # Forward pass through the policy network -------------------------
        logits = model(full_ids).logits  # (B, T, V)

        # Compute log‑probs for each *next* token -------------------------
        policy_log_probs = torch.log_softmax(logits[:, :-1], dim=-1)  # predict t+1
        target_tokens    = full_ids[:, 1:]
        logp_taken       = policy_log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        # Build a mask that is 1 for response tokens, 0 elsewhere ----------
        seq_len   = full_ids.size(1) - 1  # -1 because of the shift
        arange    = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T-1)
        start_idx = torch.tensor(start_indices, device=device).unsqueeze(1)
        mask      = arange >= start_idx  # (B, T-1)

        # Average log‑prob over the response positions --------------------
        logp_per_seq = (logp_taken * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # Always compute KL penalty for logging purposes
        with torch.no_grad():
            ref_logits = ref_model(full_ids).logits[:, :-1]
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        kl_token = torch.nn.functional.kl_div(
            policy_log_probs,
            ref_log_probs.exp(),
            reduction="none",
        ).sum(-1)  # (B, T-1)
        kl_penalty = (kl_token * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        baseline  = rewards.mean().detach()
        advantage = rewards - baseline

        # Scale loss by gradient accumulation steps to maintain the same effective learning rate
        # Only apply KL penalty to loss if coefficient > 0
        loss = -((advantage - config.kl_coefficient * kl_penalty.detach()) * logp_per_seq).mean() / gradient_accumulation_steps
        loss.backward()

        # Store metrics for logging
        accumulated_loss += loss.item() * gradient_accumulation_steps  # Scale back for logging
        accumulated_rewards.extend(rewards.tolist())
        accumulated_advantages.extend(advantage.tolist())
        accumulated_kl_penalties.extend(kl_penalty.tolist())
        accumulated_episodes.append({
            'episode': episode,
            'prompts': prompts,
            'targets': targets,
            'thinking_contents': thinking_contents,
            'contents': contents,
            'rewards': rewards.tolist(),
            'loss': loss.item() * gradient_accumulation_steps,  # Scale back for logging
            'kl_penalty_mean': kl_penalty.mean().item(),
        })

        # Perform optimizer step and logging at the end of accumulation
        if (episode + 1) % gradient_accumulation_steps == 0 or episode == config.num_episodes - 1:
            # Optional: gradient clipping (helps with exploding variance)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Zero out special‑token gradients so they stay fixed -------------
            zero_special_token_grads(model, tokenizer)
            optimizer.step()

            # Log accumulated metrics
            avg_loss = accumulated_loss / len(accumulated_episodes) if accumulated_episodes else 0.0
            avg_reward = sum(accumulated_rewards) / len(accumulated_rewards) if accumulated_rewards else 0.0
            avg_advantage = sum(accumulated_advantages) / len(accumulated_advantages) if accumulated_advantages else 0.0
            avg_kl_penalty = sum(accumulated_kl_penalties) / len(accumulated_kl_penalties) if accumulated_kl_penalties else 0.0
            avg_gen_tokens = float(mask.sum().item() / config.batch_size)

            wandb_logger.log(
                {
                    "loss": avg_loss,
                    "reward_mean": avg_reward,
                    "advantage_mean": avg_advantage,
                    "kl_penalty_mean": avg_kl_penalty,
                    "kl_penalty_scaled": (config.kl_coefficient * avg_kl_penalty),
                    "avg_gen_tokens": avg_gen_tokens,
                    "gradient_accumulation_step": episode // gradient_accumulation_steps,
                },
                step=episode,
            )

            # Log rollouts for each episode in the accumulation
            for episode_data in accumulated_episodes:
                rollout_logger.log_rollout(
                    episode=episode_data['episode'],
                    prompts=episode_data['prompts'],
                    targets=episode_data['targets'],
                    thinking_contents=episode_data['thinking_contents'],
                    contents=episode_data['contents'],
                    rewards=episode_data['rewards'],
                    loss=episode_data['loss'],
                    kl_penalty_mean=episode_data['kl_penalty_mean'],
                )

            # Reset accumulation variables
            accumulated_loss = 0.0
            accumulated_rewards = []
            accumulated_advantages = []
            accumulated_kl_penalties = []
            accumulated_episodes = []

    wandb_logger.finish()