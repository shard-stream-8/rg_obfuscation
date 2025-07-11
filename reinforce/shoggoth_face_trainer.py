# Refactored to use ShoggothFace that now loads models internally
import random
import torch
import yaml

from models.qwen3 import prepare_thinking_input
from reinforce.logit_processor import BatchThinkingTokenBudgetProcessor
from tasks.task_loader import load_task
from tasks.prompt_templates import create_custom_prompt
from reinforce.utils import zero_special_token_grads
from wandb_logger import WandbLogger
from reinforce.rollout_logger import RolloutLogger

from reinforce.trainer import Config  # reuse unchanged Config helper
from shoggoth_face import ShoggothFace


def train(config_path: str = "config.yaml") -> None:
    """Entry-point for shoggoth-face split training.
    The logic mirrors reinforce.trainer.train but handles two distinct models
    (shoggoth & face) that share a tokenizer.
    """
    with open(config_path, "r") as f:
        config = Config(yaml.safe_load(f))

    if config.shoggoth_name is None:
        raise ValueError("shoggoth_face_trainer requires 'shoggoth_name' to be set in the config")

    if getattr(config, "kl_coefficient", 0.0) > 0:
        raise NotImplementedError("KL regularisation for shoggoth/face split not implemented yet – set kl_coefficient to 0")

    # ------------------------------------------------------------------
    # RNG seeds for reproducibility
    # ------------------------------------------------------------------
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = config.device if hasattr(config, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # RL task & helpers
    # ------------------------------------------------------------------
    task = load_task(config.task_name, config.custom_verifier_path, config)
    wandb_logger   = WandbLogger(config)
    rollout_logger = RolloutLogger(config)

    # Custom / templated prompts ---------------------------------------
    custom_prompt = None
    if hasattr(config, 'custom_prompt_path') and config.custom_prompt_path:
        if config.custom_prompt_path == "registry":
            try:
                from prompts.registry import registry
                custom_prompt = registry.get_prompt(config.task_name)
            except ImportError:
                pass
        else:
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location("custom_prompt", config.custom_prompt_path)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules["custom_prompt"] = mod
            spec.loader.exec_module(mod)
            if hasattr(mod, 'prompt'):
                custom_prompt = mod.prompt

    # ------------------------------------------------------------------
    # Generation engine wrapper (loads models internally)
    # ------------------------------------------------------------------
    gen_engine = ShoggothFace(
        shoggoth_model_name=config.shoggoth_name,
        face_model_name=config.model_name,
        device=device,
        batch_size=config.batch_size,
        max_thinking_tokens=config.max_thinking_tokens,
        min_thinking_tokens=config.min_thinking_tokens,
    )

    tokenizer = gen_engine.tokenizer

    # ------------------------------------------------------------------
    # Optimiser – single AdamW over both parameter sets
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        list(gen_engine.face_model.parameters()) + list(gen_engine.shoggoth_model.parameters()),
        lr=float(config.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=1e-2,
    )

    # ------------------------------------------------------------------
    # Gradient accumulation bookkeeping
    # ------------------------------------------------------------------
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    accumulated_loss = 0.0
    accumulated_rewards = []
    accumulated_penalties = []
    accumulated_advantages = []
    accumulated_episodes = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for episode in range(config.num_episodes):
        gen_engine.face_model.eval()
        gen_engine.shoggoth_model.eval()

        # Reset logit processor state for new episode
        if hasattr(gen_engine, "logit_processor"):
            for proc in gen_engine.logit_processor:
                if hasattr(proc, "reset"):
                    proc.reset()

        # Sample batch --------------------------------------------------
        batch_indices = [random.randrange(len(task)) for _ in range(config.batch_size)]
        batch        = [task[idx] for idx in batch_indices]

        # Build prompts -------------------------------------------------
        if custom_prompt is not None:
            prompts = [
                custom_prompt(b["question"], examples=None, metadata={"task_name": config.task_name})
                if callable(custom_prompt) else custom_prompt(b["question"])
                for b in batch
            ]
        elif hasattr(config, 'prompt_template') and config.prompt_template:
            prompts = [
                create_custom_prompt(
                    original_question=b["question"],
                    task_name=config.task_name,
                    template=config.prompt_template
                ) for b in batch
            ]
        else:
            prompts = [b["question"] for b in batch]
        targets = [b["answer"] for b in batch]

        prompt_inputs = [prepare_thinking_input(tokenizer, p, enable_thinking=True) for p in prompts]

        # ------------------------------------------------------------------
        # Generation (shoggoth then face) --------------------------------
        # ------------------------------------------------------------------
        gen_dict = gen_engine.generate(
            prompt_inputs,
            max_thinking_tokens=config.max_thinking_tokens,
            max_new_tokens=config.max_new_tokens,
        )

        full_ids     = gen_dict["sequences"]          # tensor (B, T)
        think_mask   = gen_dict["think_mask"]          # (B, T-1) bool
        face_mask    = gen_dict["face_mask"]           # (B, T-1) bool
        prompt_lens  = gen_dict["prompt_lens"]         # tensor (B,)
        pad_lens     = gen_dict["pad_lens"]            # list (B,)

        # ------------------------------------------------------------------
        # Decode for reward / logging
        # ------------------------------------------------------------------
        thinking_contents, contents = [], []
        end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        start_think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]

        for idx, (seq, p_len) in enumerate(zip(full_ids, prompt_lens)):
            pad_len = pad_lens[idx]
            response_ids = seq[p_len + pad_len:].tolist()
            try:
                thinking_start = response_ids.index(start_think_token_id)
            except ValueError:
                thinking_start = None
            try:
                thinking_end_rel = response_ids.index(end_think_token_id)
                thinking_end = p_len + thinking_end_rel + 1  # absolute index after </think>
            except ValueError:
                thinking_end = None

            if thinking_start is not None and thinking_end is not None and thinking_end > p_len + thinking_start:
                thinking_ids = seq[p_len + thinking_start + 1: thinking_end - 0]  # exclude tags
                answer_ids   = seq[thinking_end:]
                thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
                answer   = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            else:
                thinking = ""
                answer   = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

            thinking_contents.append(thinking)
            contents.append(answer)

        # ------------------------------------------------------------------
        # Reward & penalty computation (reuse verifier logic) -------------
        # ------------------------------------------------------------------
        base_rewards = []
        penalties = []
        penalized_rewards = []

        for b_item, answer_str in zip(batch, contents):
            if hasattr(task, 'score_answer') and hasattr(task.score_answer, 'calculate_word_penalty'):
                verifier = task.score_answer
                base_reward = verifier.verify(answer_str, b_item)
                penalty = verifier.calculate_word_penalty(answer_str)
                penalized_reward = base_reward - penalty
            else:
                base_reward = task.score_answer(answer_str, b_item)
                penalty = 0.0
                penalized_reward = base_reward

            base_rewards.append(base_reward)
            penalties.append(penalty)
            penalized_rewards.append(penalized_reward)

        rewards = torch.tensor(penalized_rewards, dtype=torch.float32, device=device)

        # ------------------------------------------------------------------
        # Forward passes & log-prob extraction ----------------------------
        # ------------------------------------------------------------------
        shog_logits = gen_engine.shoggoth_model(full_ids).logits[:, :-1]
        face_logits = gen_engine.face_model(full_ids).logits[:, :-1]

        policy_log_probs_shog = torch.log_softmax(shog_logits, dim=-1)
        policy_log_probs_face = torch.log_softmax(face_logits, dim=-1)

        target_tokens = full_ids[:, 1:]
        logp_taken_shog = policy_log_probs_shog.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        logp_taken_face = policy_log_probs_face.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)

        # Average log-prob over designated positions
        logp_shog_seq = (logp_taken_shog * think_mask).sum(dim=1) / (think_mask.sum(dim=1) + 1e-8)
        logp_face_seq = (logp_taken_face * face_mask).sum(dim=1) / (face_mask.sum(dim=1) + 1e-8)
        logp_per_seq  = logp_shog_seq + logp_face_seq

        # ------------------------------------------------------------------
        # Policy-gradient loss -------------------------------------------
        # ------------------------------------------------------------------
        baseline  = rewards.mean().detach()
        advantage = rewards - baseline

        loss = -(advantage * logp_per_seq).mean() / gradient_accumulation_steps
        loss.backward()

        # ------------------------------------------------------------------
        # Bookkeeping for accumulation -----------------------------------
        # ------------------------------------------------------------------
        accumulated_loss += loss.item() * gradient_accumulation_steps
        accumulated_rewards.extend(rewards.tolist())
        accumulated_penalties.extend(penalties)
        accumulated_advantages.extend(advantage.tolist())
        accumulated_episodes.append({
            'episode': episode,
            'prompts': prompts,
            'targets': targets,
            'thinking_contents': thinking_contents,
            'contents': contents,
            'rewards': rewards.tolist(),
            'penalties': penalties,
            'loss': loss.item() * gradient_accumulation_steps,
        })

        # ------------------------------------------------------------------
        # Optimiser step & logging ---------------------------------------
        # ------------------------------------------------------------------
        if (episode + 1) % gradient_accumulation_steps == 0 or episode == config.num_episodes - 1:
            torch.nn.utils.clip_grad_norm_(list(gen_engine.face_model.parameters()) + list(gen_engine.shoggoth_model.parameters()), 1.0)

            zero_special_token_grads(gen_engine.face_model, tokenizer)
            zero_special_token_grads(gen_engine.shoggoth_model, tokenizer)
            optimizer.step()
            optimizer.zero_grad()

            # Aggregate metrics
            avg_loss = accumulated_loss / len(accumulated_episodes)
            avg_reward = sum(accumulated_rewards) / len(accumulated_rewards)
            avg_penalty = sum(accumulated_penalties) / len(accumulated_penalties)
            avg_advantage = sum(accumulated_advantages) / len(accumulated_advantages)

            wandb_logger.log({
                "loss": avg_loss,
                "penalized_reward": avg_reward,
                "penalty": avg_penalty,
                "advantage_mean": avg_advantage,
                "gradient_accumulation_step": episode // gradient_accumulation_steps,
            }, step=episode)

            # Rollout logs
            for ep in accumulated_episodes:
                rollout_logger.log_rollout(
                    episode=ep['episode'],
                    prompts=ep['prompts'],
                    targets=ep['targets'],
                    thinking_contents=ep['thinking_contents'],
                    contents=ep['contents'],
                    rewards=ep['rewards'],
                    loss=ep['loss'],
                )

            # Reset accumulators
            accumulated_loss = 0.0
            accumulated_rewards = []
            accumulated_penalties = []
            accumulated_advantages = []
            accumulated_episodes = []

    wandb_logger.finish() 