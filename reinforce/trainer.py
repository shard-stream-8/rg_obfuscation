import torch
import yaml
from models.qwen3 import load_qwen3_model, prepare_thinking_input
from reinforce.logit_processor import BatchThinkingTokenBudgetProcessor
from reinforce.kl_penalty import compute_kl_penalty
from reinforce.utils import zero_special_token_grads
from tasks.task_loader import load_task
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

def train(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = Config(yaml.safe_load(f))
    torch.manual_seed(config.seed)
    model, tokenizer, device = load_qwen3_model(config.model_name, config.device)
    ref_model = None
    if config.use_kl_penalty:
        ref_model, _, _ = load_qwen3_model(config.model_name, config.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    task = load_task(config.task_name, config.custom_verifier_path)
    wandb_logger = WandbLogger(config)
    rollout_logger = RolloutLogger(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate))
    logit_processor = BatchThinkingTokenBudgetProcessor(
        tokenizer,
        max_thinking_tokens=config.max_thinking_tokens,
        batch_size=config.batch_size,
        min_thinking_tokens=config.min_thinking_tokens
    )
    for episode in range(config.num_episodes):
        # Sample random indices for the batch
        import random
        batch_indices = [random.randint(0, len(task) - 1) for _ in range(config.batch_size)]
        batch = [task[idx] for idx in batch_indices]
        prompts = [b["question"] for b in batch]
        targets = [b["answer"] for b in batch]
        model_inputs = tokenizer([
            prepare_thinking_input(tokenizer, p, enable_thinking=True) for p in prompts
        ], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=config.max_new_tokens,
                logits_processor=[logit_processor],
                return_dict_in_generate=True,
                output_scores=True
            )
        generated_ids = outputs.sequences
        # Parse thinking and content as in user prompt
        output_ids = [g[len(inp):].tolist() for g, inp in zip(generated_ids, model_inputs.input_ids)]
        thinking_contents, contents = [], []
        for out in output_ids:
            try:
                index = len(out) - out[::-1].index(tokenizer.encode("</think>", add_special_tokens=False)[0])
            except ValueError:
                index = 0
            thinking_contents.append(tokenizer.decode(out[:index], skip_special_tokens=True).strip("\n"))
            contents.append(tokenizer.decode(out[index:], skip_special_tokens=True).strip("\n"))
        # Compute rewards using task score_answer method
        rewards = torch.tensor([task.score_answer(c, b) for b, c in zip(batch, contents)], dtype=torch.float32, device=device)
        
        # Compute advantage based on batch mean reward
        advantage = rewards - rewards.mean()
        
        # Compute policy loss (REINFORCE)
        model.train()
        optimizer.zero_grad()
        # Forward pass for logprobs
        full_ids = tokenizer([
            prepare_thinking_input(tokenizer, p, enable_thinking=True) for p in prompts
        ], return_tensors="pt", padding=True).input_ids.to(device)
        outputs = model(full_ids, labels=full_ids)
        logits = outputs.logits
        # Assume response starts after prompt
        response_start_idx = model_inputs.input_ids.shape[1]
        response_logits = logits[:, response_start_idx:-1]
        # Compute logprobs for actions
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        # Mask for valid tokens (not padding)
        gradient_mask = (full_ids[:, response_start_idx+1:] != tokenizer.pad_token_id).float()
        # Policy gradient loss
        log_probs_taken = log_probs.gather(-1, full_ids[:, response_start_idx+1:].unsqueeze(-1)).squeeze(-1)
        log_probs_taken = (log_probs_taken * gradient_mask).sum(dim=-1) / (gradient_mask.sum(dim=-1) + 1e-8)
        # KL penalty
        kl_penalty = compute_kl_penalty(rewards, ref_model, config, full_ids, response_start_idx, response_logits, gradient_mask)
        loss = -((advantage - kl_penalty) * log_probs_taken).mean()
        loss.backward()
        zero_special_token_grads(model, tokenizer)
        optimizer.step()
        # Logging
        wandb_logger.log({
            "loss": loss.item(),
            "reward_mean": rewards.mean().item(),
            "advantage_mean": advantage.mean().item(),
            "kl_penalty_mean": kl_penalty.mean().item(),
            "episode": episode
        }, step=episode)
        
        # Log rollout to JSON
        rollout_logger.log_rollout(
            episode=episode,
            prompts=prompts,
            targets=targets,
            thinking_contents=thinking_contents,
            contents=contents,
            rewards=rewards.tolist(),
            loss=loss.item(),
            kl_penalty_mean=kl_penalty.mean().item()
        )
    wandb_logger.finish() 