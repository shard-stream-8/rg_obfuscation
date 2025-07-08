import random
import torch
import yaml

from models.qwen3 import load_qwen3_model, prepare_thinking_input
from reinforce.logit_processor import BatchThinkingTokenBudgetProcessor
from tasks.task_loader import load_task
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
    if config.use_kl_penalty:
        ref_model, _, _ = load_qwen3_model(config.model_name, config.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # RL task & logging helpers -------------------------------------------
    task = load_task(config.task_name, config.custom_verifier_path)
    wandb_logger   = WandbLogger(config)
    rollout_logger = RolloutLogger(config)

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

    for episode in range(config.num_episodes):
        model.eval()

        batch_indices = [random.randrange(len(task)) for _ in range(config.batch_size)]
        batch        = [task[idx] for idx in batch_indices]
        prompts      = [b["question"] for b in batch]
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
        for seq, p_len in zip(generated_ids, prompt_lens):
            response_ids = seq[p_len:].tolist()
            try:
                # last occurrence of </think>
                idx = len(response_ids) - response_ids[::-1].index(end_think_token_id)
            except ValueError:
                idx = 0
            thinking = tokenizer.decode(response_ids[:idx],  skip_special_tokens=True).strip("\n")
            answer   = tokenizer.decode(response_ids[idx:], skip_special_tokens=True).strip("\n")
            thinking_contents.append(thinking)
            contents.append(answer)

        rewards = torch.tensor(
            [task.score_answer(c, b) for b, c in zip(batch, contents)],
            dtype=torch.float32,
            device=device,
        )

        model.train()
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

        if config.use_kl_penalty:
            with torch.no_grad():
                ref_logits = ref_model(full_ids).logits[:, :-1]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            kl_token = torch.nn.functional.kl_div(
                policy_log_probs,
                ref_log_probs.exp(),
                reduction="none",
            ).sum(-1)  # (B, T-1)
            kl_penalty = (kl_token * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            kl_penalty = torch.zeros_like(rewards)

        baseline  = rewards.mean().detach()
        advantage = rewards - baseline

        loss = -((advantage - config.kl_coefficient * kl_penalty.detach()) * logp_per_seq).mean()
        loss.backward()

        # Optional: gradient clipping (helps with exploding variance)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Zero out special‑token gradients so they stay fixed -------------
        zero_special_token_grads(model, tokenizer)
        optimizer.step()

        wandb_logger.log(
            {
                "loss":                loss.item(),
                "reward_mean":         rewards.mean().item(),
                "advantage_mean":      advantage.mean().item(),
                "kl_penalty_mean":     kl_penalty.mean().item(),
                "kl_penalty_scaled":  (config.kl_coefficient * kl_penalty).mean().item(),
                "avg_gen_tokens":     float(mask.sum().item() / config.batch_size),
            },
            step=episode,
        )

        rollout_logger.log_rollout(
            episode=episode,
            prompts=prompts,
            targets=targets,
            thinking_contents=thinking_contents,
            contents=contents,
            rewards=rewards.tolist(),
            loss=loss.item(),
            kl_penalty_mean=kl_penalty.mean().item(),
        )

    wandb_logger.finish()