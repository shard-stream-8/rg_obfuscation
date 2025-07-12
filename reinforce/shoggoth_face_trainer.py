# Refactored to use ShoggothFace that now loads models internally
import random
import torch
import yaml
import gc

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
    # Newly tracked accumulators to match trainer.py logging
    accumulated_base_rewards = []
    accumulated_thinking_penalties = []
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
        try:
            gen_dict = gen_engine.generate(
                prompt_inputs,
                max_thinking_tokens=config.max_thinking_tokens,
                max_new_tokens=config.max_new_tokens,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
            if isinstance(oom_err, torch.cuda.OutOfMemoryError) or "out of memory" in str(oom_err).lower():
                print(f"Episode {episode}: CUDA OOM during generation. Skipping episode and clearing cache.")
                for _var in ['gen_dict', 'full_ids']:
                    if _var in locals():
                        del locals()[_var]
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise

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
        thinking_penalties = []  # per-example thinking penalties for logging
        penalized_rewards = []
        # Word-level penalty bookkeeping
        output_word_penalties = []
        thinking_word_penalties = []
        output_word_counts = []
        thinking_word_counts = []

        for b_item, answer_str, think_str in zip(batch, contents, thinking_contents):
            if hasattr(task, 'score_answer') and hasattr(task.score_answer, 'calculate_word_penalty'):
                verifier = task.score_answer
                base_reward = verifier.verify(answer_str, b_item)
                penalty = verifier.calculate_word_penalty(answer_str)

                # Detailed logging (only if verifier implements helpers)
                thinking_penalty_for_logging = 0.0
                output_penalty_for_logging = 0.0
                output_word_penalties_dict = {}
                thinking_word_penalties_dict = {}
                output_word_counts_dict = {}
                thinking_word_counts_dict = {}

                if hasattr(verifier, 'calculate_word_penalty_for_logging'):
                    output_penalty_for_logging = verifier.calculate_word_penalty_for_logging(answer_str)
                if hasattr(verifier, 'calculate_thinking_penalty_for_logging'):
                    thinking_penalty_for_logging = verifier.calculate_thinking_penalty_for_logging(think_str)
                if hasattr(verifier, 'calculate_individual_word_penalties'):
                    output_word_penalties_dict = verifier.calculate_individual_word_penalties(answer_str)
                if hasattr(verifier, 'calculate_individual_thinking_word_penalties'):
                    thinking_word_penalties_dict = verifier.calculate_individual_thinking_word_penalties(think_str)
                if hasattr(verifier, 'calculate_raw_word_counts'):
                    output_word_counts_dict = verifier.calculate_raw_word_counts(answer_str)
                if hasattr(verifier, 'calculate_raw_thinking_word_counts'):
                    thinking_word_counts_dict = verifier.calculate_raw_thinking_word_counts(think_str)

                penalized_reward = base_reward - penalty
            else:
                base_reward = task.score_answer(answer_str, b_item)
                penalty = 0.0
                thinking_penalty_for_logging = 0.0
                output_penalty_for_logging = 0.0
                output_word_penalties_dict = {}
                thinking_word_penalties_dict = {}
                output_word_counts_dict = {}
                thinking_word_counts_dict = {}
                penalized_reward = base_reward

            base_rewards.append(base_reward)
            penalties.append(penalty)
            thinking_penalties.append(thinking_penalty_for_logging)
            penalized_rewards.append(penalized_reward)
            output_word_penalties.append(output_word_penalties_dict)
            thinking_word_penalties.append(thinking_word_penalties_dict)
            output_word_counts.append(output_word_counts_dict)
            thinking_word_counts.append(thinking_word_counts_dict)

        rewards = torch.tensor(penalized_rewards, dtype=torch.float32, device=device)

        # ------------------------------------------------------------------
        # Forward passes & log-prob extraction ----------------------------
        # ------------------------------------------------------------------
        try:
            shog_logits = gen_engine.shoggoth_model(full_ids).logits[:, :-1]
            face_logits = gen_engine.face_model(full_ids).logits[:, :-1]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
            if isinstance(oom_err, torch.cuda.OutOfMemoryError) or "out of memory" in str(oom_err).lower():
                print(f"Episode {episode}: CUDA OOM during forward pass. Skipping episode and clearing cache.")
                for _var in ['shog_logits', 'face_logits', 'logp_taken_shog', 'logp_taken_face', 'full_ids']:
                    if _var in locals():
                        del locals()[_var]
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise

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
        try:
            loss.backward()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
            if isinstance(oom_err, torch.cuda.OutOfMemoryError) or "out of memory" in str(oom_err).lower():
                print(f"Episode {episode}: CUDA OOM during backward pass. Skipping episode and clearing cache.")
                if 'loss' in locals():
                    del loss
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise

        # ------------------------------------------------------------------
        # Bookkeeping for accumulation -----------------------------------
        # ------------------------------------------------------------------
        accumulated_loss += loss.item() * gradient_accumulation_steps
        accumulated_rewards.extend(rewards.tolist())
        accumulated_penalties.extend(penalties)
        accumulated_advantages.extend(advantage.tolist())
        accumulated_base_rewards.extend(base_rewards)
        accumulated_thinking_penalties.extend(thinking_penalties)
        accumulated_episodes.append({
            'episode': episode,
            'prompts': prompts,
            'targets': targets,
            'thinking_contents': thinking_contents,
            'contents': contents,
            'rewards': rewards.tolist(),
            'base_rewards': base_rewards,
            'penalties': penalties,
            'thinking_penalties': thinking_penalties,
            'output_word_penalties': output_word_penalties,
            'thinking_word_penalties': thinking_word_penalties,
            'output_word_counts': output_word_counts,
            'thinking_word_counts': thinking_word_counts,
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
            avg_base_reward = sum(accumulated_base_rewards) / len(accumulated_base_rewards) if accumulated_base_rewards else 0.0
            avg_thinking_penalty = sum(accumulated_thinking_penalties) / len(accumulated_thinking_penalties) if accumulated_thinking_penalties else 0.0
            avg_advantage = sum(accumulated_advantages) / len(accumulated_advantages)

            # Average generated tokens (think + face masks)
            combined_mask = (think_mask | face_mask)
            avg_gen_tokens = float(combined_mask.sum().item() / config.batch_size)

            # ------------------------------------------------------------------
            # Aggregate word-level penalties/counts across accumulation --------
            # ------------------------------------------------------------------
            all_output_word_penalties = {}
            all_thinking_word_penalties = {}
            all_output_word_counts = {}
            all_thinking_word_counts = {}
            for ep in accumulated_episodes:
                for wp in ep['output_word_penalties']:
                    for w, p_val in wp.items():
                        all_output_word_penalties.setdefault(w, []).append(p_val)
                for wp in ep['thinking_word_penalties']:
                    for w, p_val in wp.items():
                        all_thinking_word_penalties.setdefault(w, []).append(p_val)
                for wc in ep['output_word_counts']:
                    for w, cnt in wc.items():
                        all_output_word_counts.setdefault(w, []).append(cnt)
                for wc in ep['thinking_word_counts']:
                    for w, cnt in wc.items():
                        all_thinking_word_counts.setdefault(w, []).append(cnt)

            avg_output_word_penalties = {f"output_penalty_{w}": sum(v)/len(v) for w, v in all_output_word_penalties.items()}
            avg_thinking_word_penalties = {f"thinking_penalty_{w}": sum(v)/len(v) for w, v in all_thinking_word_penalties.items()}
            avg_output_word_counts = {f"output_count_{w}": sum(v)/len(v) for w, v in all_output_word_counts.items()}
            avg_thinking_word_counts = {f"thinking_count_{w}": sum(v)/len(v) for w, v in all_thinking_word_counts.items()}

            total_output_words = sum(sum(v) for v in all_output_word_counts.values())
            total_thinking_words = sum(sum(v) for v in all_thinking_word_counts.values())

            wandb_logger.log({
                "loss": avg_loss,
                "reward_mean": avg_base_reward,
                "penalized_reward": avg_reward,
                "penalty": avg_penalty,
                "thinking_penalty_mean": avg_thinking_penalty,
                "advantage_mean": avg_advantage,
                "avg_gen_tokens": avg_gen_tokens,
                "gradient_accumulation_step": episode // gradient_accumulation_steps,
                "total_output_penalized_words": total_output_words,
                "total_thinking_penalized_words": total_thinking_words,
                **avg_output_word_penalties,
                **avg_thinking_word_penalties,
                **avg_output_word_counts,
                **avg_thinking_word_counts,
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
            accumulated_base_rewards = []
            accumulated_thinking_penalties = []
            accumulated_episodes = []

    wandb_logger.finish() 