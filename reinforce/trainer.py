import torch
import torch.nn.functional as F
import random
import yaml
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from wandb_logger import WandbLogger
from .rollout_logger import RolloutLogger
from .logit_processor import BatchThinkingTokenBudgetProcessor
from .utils import zero_special_token_grads
from .judge import JudgePenalty, RegexPenalty
from prompts.terminal_prompts import (
    get_multi_turn_terminal_instructions,
    NO_COMMAND_MESSAGE,
    get_verifier_incorrect_message,
    get_normal_terminal_message,
    get_command_failed_message
)
from tasks.task_loader import load_task
from prompts.registry import registry
from tasks.prompt_templates import create_custom_prompt
from models.qwen3 import load_qwen3_model, prepare_thinking_input
import asyncio

class Config:
    def __init__(self, d):
        self.__dict__.update(d)
        self._ensure_defaults()

    def __getitem__(self, k):
        return self.__dict__[k]

    def __iter__(self):
        return iter(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def __contains__(self, k):
        return k in self.__dict__

    def get(self, k, default=None):
        return self.__dict__.get(k, default)

    def _ensure_defaults(self):
        # Add any default values here
        # Ensure the wrong answer flag always exists so downstream code can rely on it
        if 'wrong_answer' not in self.__dict__:
            self.wrong_answer = False
        # Optional HF repo for a separate shoggoth (CoT) model
        if 'shoggoth_name' not in self.__dict__:
            self.shoggoth_name = None
        # Option to save rollouts to W&B as a jsonl artifact
        if 'save_rollouts_to_wandb' not in self.__dict__:
            self.save_rollouts_to_wandb = False
        # If True, chain-of-thought gradients are severed as described in RL loss
        if 'sever_gradients' not in self.__dict__:
            self.sever_gradients = False
        pass

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file with inheritance support."""
    # Load base config first
    base_config = {}
    base_path = "configs/base.yaml"
    if os.path.exists(base_path):
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
    
    # Load specific config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Merge configs (specific config overrides base config)
    merged_config = {**base_config, **config_dict}
    return Config(merged_config)

def load_model_and_tokenizer(config: Config):
    """Load face model (and optionally shoggoth model) plus tokenizer & device."""
    face_model, tokenizer, device = load_qwen3_model(config.model_name, config.device)

    shoggoth_model = None
    if getattr(config, 'shoggoth_name', None):
        shoggoth_model, shog_tok, _ = load_qwen3_model(config.shoggoth_name, config.device)
        # Sanity check: identical vocabularies
        if tokenizer.get_vocab() != shog_tok.get_vocab():
            raise ValueError("Tokenizers of face and shoggoth models differ – they must share the same vocabulary")

    return face_model, tokenizer, device, shoggoth_model

def setup_task_and_logging(config: Config) -> Tuple[Any, Any]:
    """Setup task and logging components."""
    task = load_task(config.task_name, config.custom_verifier_path, config)
    wandb_logger = WandbLogger(config)
    rollout_logger = RolloutLogger(config)
    return task, wandb_logger, rollout_logger

def setup_training_components(config: Config, face_model: Any, tokenizer: Any, shoggoth_model: Any = None):
    """Setup optimizer and logit processor."""
    params = list(face_model.parameters())
    if shoggoth_model is not None:
        params += list(shoggoth_model.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=float(config.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=1e-2,
    )

    logit_processor = BatchThinkingTokenBudgetProcessor(
        tokenizer,
        max_thinking_tokens=config.max_thinking_tokens,
        batch_size=config.batch_size,
        min_thinking_tokens=config.min_thinking_tokens,
    )

    return optimizer, logit_processor

def load_custom_prompt(config: Config) -> Optional[Any]:
    """Load custom prompt from registry or file."""
    if not hasattr(config, 'custom_prompt_path') or not config.custom_prompt_path:
        return None
    if config.custom_prompt_path == "registry":
        try:
            use_terminal = getattr(config, 'use_terminal', False)
            custom_prompt = registry.get_prompt(config.task_name, use_terminal=use_terminal)
            if custom_prompt is not None:
                print(f"Loaded custom prompt for task '{config.task_name}' from registry")
            else:
                print(f"No custom prompt found in registry for task '{config.task_name}', using default")
            return custom_prompt
        except ImportError:
            print("Prompt registry not available, falling back to default prompt")
            return None
    else:
        # Use the old file-based approach
        return registry.load_prompt_from_file(config.custom_prompt_path)

def build_prompts(config: Config, batch: List[Dict], custom_prompt: Optional[Any]) -> List[str]:
    """Build prompts for the batch."""
    if custom_prompt is not None:
        use_terminal = getattr(config, 'use_terminal', False)
        prompts = [
            custom_prompt(b["question"], examples=None, metadata={
                "task_name": config.task_name,
                "use_terminal": use_terminal
            }) if callable(custom_prompt) else custom_prompt(b["question"]) for b in batch
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

    # Prepend terminal instructions if needed
    if getattr(config, 'use_terminal', False):
        prompts = [get_multi_turn_terminal_instructions(p) for p in prompts]
   
    return prompts


def extract_thinking_and_content(tokenizer: Any, response_ids: List[int]) -> Tuple[str, str]:
    """Extract thinking and content from model response."""
    end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    start_think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]

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
            thinking_ids = response_ids[thinking_start:thinking_end]
            content_ids = response_ids[thinking_end:]
            thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
            content = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
        else:
            thinking_ids = response_ids[thinking_start:]
            thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
            content = ""
    else:
        thinking = ""
        content = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Clean up prompt fragments
    prompt_fragments = [
        "Write in as much detail as is useful inside think tags, but give only a brief explanation in your final output.",
        "assistant",
        "user:",
        "assistant:"
    ]
    for fragment in prompt_fragments:
        thinking = thinking.replace(fragment, "").strip()
        content = content.replace(fragment, "").strip()

    return thinking, content

def calculate_rewards_and_penalties(task: Any, batch: List[Dict], contents: List[str],
                                  thinking_contents: List[str], judge_penalty: JudgePenalty = None, regex_penalty: RegexPenalty = None) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Calculate rewards and penalties for the batch."""
    base_rewards, penalties, thinking_penalties, penalized_rewards, judge_penalties, judge_scores, regex_penalties = [], [], [], [], [], [], []

    for b, c, thinking_c in zip(batch, contents, thinking_contents):
        if hasattr(task, 'process_model_output'):
            # Terminal task
            result = task.process_model_output(c, b['answer'])
            base_reward = result['reward']
            penalty = 0.0
            thinking_penalty = 0.0
            penalized_reward = base_reward
        else:
            # Regular task
            if hasattr(task, 'score_answer') and hasattr(task.score_answer, 'calculate_word_penalty'):
                verifier = task.score_answer
                base_reward = verifier.verify(c, b)
                penalty = verifier.calculate_word_penalty(c)
                thinking_penalty = 0.0
                penalized_reward = base_reward - penalty
            else:
                base_reward = task.score_answer(c, b)
                penalty = 0.0
                thinking_penalty = 0.0
                penalized_reward = base_reward

        # Add judge penalty if enabled
        judge_penalty_value = 0.0
        judge_score = 0.0
        if judge_penalty is not None:
            judge_penalty_value, judge_score = judge_penalty.calculate_penalty_sync_with_score(b['question'], c, None)
            penalized_reward -= judge_penalty_value

        # Add regex penalty if enabled
        regex_penalty_value = 0.0
        if regex_penalty is not None:
            regex_penalty_value, _ = regex_penalty.calculate_penalty(b['question'], c)
            penalized_reward -= regex_penalty_value

        base_rewards.append(base_reward)
        penalties.append(penalty)
        thinking_penalties.append(thinking_penalty)
        judge_penalties.append(judge_penalty_value)
        judge_scores.append(judge_score)
        regex_penalties.append(regex_penalty_value)
        penalized_rewards.append(penalized_reward)

    return base_rewards, penalties, thinking_penalties, penalized_rewards, judge_penalties, judge_scores, regex_penalties

def perform_training_step(model: Any, optimizer: Any, total_advantage: torch.Tensor,
                        task_advantage: Optional[torch.Tensor], logits: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor,
                        think_mask: Optional[torch.Tensor], config: Config, gradient_accumulation_steps: int, input_ids: torch.Tensor = None) -> Tuple[float, float]:
    """Perform the actual training step."""
    model.train()

    # Compute log-probs ---------------------------------------------------
    policy_log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    logp_taken = policy_log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)

    # --------------------------------------------------------------------
    # Choose loss formulation based on sever_gradients flag
    # --------------------------------------------------------------------
    if not getattr(config, "sever_gradients", False):
        # Legacy path -----------------------------------------------------
        logp_per_seq = (logp_taken * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        if total_advantage.dim() == 0 or logp_per_seq.dim() == 0 or total_advantage.numel() == 0 or logp_per_seq.numel() == 0:
            return 0

        loss = -(total_advantage * logp_per_seq).mean() / gradient_accumulation_steps
        loss.backward()
        return loss.item() * gradient_accumulation_steps

    # New sever_gradients path ------------------------------------------
    assert think_mask is not None, "think_mask must be provided when sever_gradients is True"
    assert task_advantage is not None, "task_advantage must be provided when sever_gradients is True"

    # Separate thinking vs output masks (ensure same dtype)
    think_mask_bool = think_mask.bool() & mask
    output_mask_bool = (~think_mask_bool) & mask

    # Handle sequences with zero tokens in a segment gracefully
    def _safe_avg(numer, denom_mask):
        denom = denom_mask.sum(dim=1)
        return (numer * denom_mask).sum(dim=1) / (denom + 1e-8)

    logp_thinking = _safe_avg(logp_taken, think_mask_bool)
    logp_output   = _safe_avg(logp_taken, output_mask_bool)

    loss_seq = -(total_advantage * logp_output + task_advantage * logp_thinking)
    loss = loss_seq.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.item() * gradient_accumulation_steps

# ============================================================
# Dual-model training step (face + shoggoth)
# ============================================================

def perform_training_step_dual(face_model: Any, shoggoth_model: Any, optimizer: Any, total_advantage: torch.Tensor,
                               task_advantage: Optional[torch.Tensor], target_tokens: torch.Tensor, mask: torch.Tensor, think_mask: torch.Tensor,
                               gradient_accumulation_steps: int, input_ids: torch.Tensor, config: Config) -> float:
    """Training step that mixes logits from face and shoggoth models.

    The `think_mask` tells which positions (whose NEXT token) were generated by the shoggoth model.
    We compute log-probs from both models and select per-token values accordingly.
    """

    face_model.train()
    shoggoth_model.train()

    # Forward passes -----------------------------------------------------
    logits_face = face_model(input_ids).logits  # [B, T, V]
    logits_shog = shoggoth_model(input_ids).logits

    # Log-probs -----------------------------------------------------------
    logp_face = torch.log_softmax(logits_face[:, :-1], dim=-1)
    logp_shog = torch.log_softmax(logits_shog[:, :-1], dim=-1)

    # Gather log-prob of actual taken tokens
    logp_face_taken = logp_face.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
    logp_shog_taken = logp_shog.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)

    think_mask_bool = think_mask.bool()

    # --------------------------------------------------------------------
    # Choose loss path depending on sever_gradients
    # --------------------------------------------------------------------
    if not getattr(config, "sever_gradients", False):
        # Mixed logp from selected model per token
        combined_logp_taken = torch.where(think_mask_bool, logp_shog_taken, logp_face_taken)
        logp_per_seq = (combined_logp_taken * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        if total_advantage.dim() == 0 or logp_per_seq.dim() == 0 or total_advantage.numel() == 0 or logp_per_seq.numel() == 0:
            return 0.0

        loss = -(total_advantage * logp_per_seq).mean() / gradient_accumulation_steps
        loss.backward()
        return loss.item() * gradient_accumulation_steps

    # Sever gradients path ----------------------------------------------
    assert task_advantage is not None, "task_advantage must be provided when sever_gradients is True"

    # Separate averages for thinking (shoggoth) and output (face)
    def _safe_avg(values, mask_part):
        denom = mask_part.sum(dim=1)
        return (values * mask_part).sum(dim=1) / (denom + 1e-8)

    mask_thinking = think_mask_bool & mask
    mask_output = (~think_mask_bool) & mask

    logp_thinking = _safe_avg(logp_shog_taken, mask_thinking)
    logp_output = _safe_avg(logp_face_taken, mask_output)

    loss_seq = -(total_advantage * logp_output + task_advantage * logp_thinking)
    loss = loss_seq.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.item() * gradient_accumulation_steps

def run_batched_multi_turn_episodes(face_model: Any, tokenizer: Any, task: Any, initial_prompts: List[str],
                                   logit_processor: Any, config: Config, device: str, ground_truths: List[Any],
                                   shoggoth_model: Any = None) -> List[Dict[str, Any]]:
    """
    Run multiple multi-turn episodes in parallel (batched by turn).
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        task: The task instance
        initial_prompts: List of initial prompts for each episode
        logit_processor: The logit processor
        config: Configuration object
        device: Device to run on
        ground_truths: List of ground truth data for each episode
        
    Returns:
        List of episode results
    """
    batch_size = len(initial_prompts)
    max_turns = getattr(config, 'max_turns', 10)
    
    # Initialize episode states for all episodes
    episode_states = [task.create_episode_state() for _ in range(batch_size)]
    episode_results = [{
        'episode_rewards': [],
        'episode_commands': [],
        'episode_outputs': [],
        'episode_thinking_contents': [],
        'episode_contents': [],
        'conversation_dialogue': [{"role": "human", "content": prompt}],
        'final_reward': 0.0,
        'episode_complete': False,
        'turn_count': 0,
        'terminal_context': '',
        # Store training data for each turn
        'turn_target_tokens': [],
        'turn_masks': [],
        'turn_input_ids': [],
        'turn_think_masks': [],  # boolean masks for CoT tokens when shoggoth_model is used
    } for prompt in initial_prompts]

    # If a separate shoggoth model is provided, construct the wrapper once
    shoggoth_face = None
    if shoggoth_model is not None:
        from shoggoth_face import ShoggothFace  # local import to avoid circularity if any
        shoggoth_face = ShoggothFace(
            shoggoth_model_name=None,
            face_model_name=None,
            shoggoth_model=shoggoth_model,
            face_model=face_model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_thinking_tokens=config.max_thinking_tokens,
            min_thinking_tokens=config.min_thinking_tokens,
            logit_processor=logit_processor,
        )

    
    # Track which episodes are still active
    active_episodes = list(range(batch_size))
    
    for turn in range(max_turns):
        if not active_episodes:
            break
            
        logit_processor.reset()
        
        # Prepare prompts for active episodes
        active_prompts = []
        active_indices = []
        
        for episode_idx in active_episodes:
            episode_result = episode_results[episode_idx]
            
            if turn == 0:
                # First turn: use initial prompt
                current_prompt = initial_prompts[episode_idx]
                prompt_input = prepare_thinking_input(tokenizer, current_prompt, enable_thinking=True)
            else:
                # Subsequent turns: build conversation
                conversation_text = ""
                for msg in episode_result['conversation_dialogue']:
                    if msg['role'] == 'human':
                        conversation_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                    elif msg['role'] == 'assistant':
                        conversation_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
                
                conversation_text += "<|im_start|>assistant\n"
                prompt_input = conversation_text
            
            active_prompts.append(prompt_input)
            active_indices.append(episode_idx)

        # Generate responses for all active episodes --------------------------------
        if shoggoth_face is None:
            model_input = tokenizer(active_prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = face_model.generate(
                    **model_input,
                    max_new_tokens=config.max_new_tokens,
                    logits_processor=[logit_processor],
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            generated_ids = outputs.sequences
            think_mask_batch = None  # not available
            # (Debug prints removed per latest instructions)
        else:
            with torch.no_grad():
                sof_outputs = shoggoth_face.generate(
                    active_prompts,
                    max_thinking_tokens=config.max_thinking_tokens,
                    max_new_tokens=config.max_new_tokens,
                )
            generated_ids = sof_outputs["sequences"]
            think_mask_batch = sof_outputs["think_mask"]  # (B, seq_len)

            # (Debug prints removed per latest instructions)

        prompt_len = model_input.input_ids.shape[1] - 2 if shoggoth_face is None else None
                
        # ------------------------------------------------------------------
        # Construct loss mask
        # ------------------------------------------------------------------
        seq_len = generated_ids.size(1) - 1  # logits/targets length
        arange = torch.arange(seq_len, device=device).unsqueeze(0)

        # Baseline mask: positions belonging to the assistant response only
        if shoggoth_face is None:
            baseline_mask = arange >= prompt_len  # shape [B, seq_len]
        else:
            # Use per-sample prompt lengths provided by the shoggoth_face output
            raw_p_lens = sof_outputs["prompt_lens"]  # tensor or list
            if isinstance(raw_p_lens, torch.Tensor):
                prompt_lens_tensor = raw_p_lens.to(device)
            else:
                prompt_lens_tensor = torch.tensor(raw_p_lens, device=device)
            baseline_mask = arange >= prompt_lens_tensor.unsqueeze(1)  # broadcast to [B, seq_len]

        # Padding mask: exclude tokens that are padding so they never contribute
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
        nonpad_mask = (generated_ids[:, 1:] != pad_id)  # shape [B, seq_len]

        # Combine masks: always train on assistant tokens only (user tokens excluded)
        mask = baseline_mask & nonpad_mask  # assistant tokens only, no pads

        # ------------------------------------------------------------------
        # Derive think_mask for single-model path if needed
        # ------------------------------------------------------------------
        if think_mask_batch is None:
            # Build a boolean mask marking positions whose *next* token is part
            # of the <think>…</think> chain-of-thought.  This is needed for the
            # sever_gradients option.
            start_think_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
            end_think_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

            think_mask_list = []
            for i in range(generated_ids.size(0)):
                seq = generated_ids[i]
                seq_len_plus1 = seq.size(0)
                seq_len = seq_len_plus1 - 1  # logits length
                t_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

                inside = False
                # Iterate over token positions whose logits contributed to the
                # NEXT token (hence up to seq_len-1)
                for pos in range(prompt_len, seq_len_plus1 - 1):
                    token_id = seq[pos].item()
                    if token_id == start_think_id:
                        inside = True
                        # The NEXT token will be inside CoT; current pos is tag
                        continue
                    if token_id == end_think_id:
                        inside = False
                    if inside:
                        t_mask[pos] = True
                think_mask_list.append(t_mask)

            think_mask_batch = torch.stack(think_mask_list, dim=0)

         
        # Store training data for each episode
        for i, episode_idx in enumerate(active_indices):
            episode_results[episode_idx]['turn_target_tokens'].append(generated_ids[i:i+1, 1:])
            episode_results[episode_idx]['turn_masks'].append(mask[i:i+1])
            episode_results[episode_idx]['turn_input_ids'].append(generated_ids[i:i+1])
            episode_results[episode_idx]['turn_think_masks'].append(think_mask_batch[i:i+1])

        
        # Process each active episode
        completed_episodes = set()
        
        for i, episode_idx in enumerate(active_indices):
            if shoggoth_face is None:
                response_ids = generated_ids[i][prompt_len:].tolist()
            else:
                p_len_i = sof_outputs["prompt_lens"][i].item()
                response_ids = generated_ids[i][p_len_i:].tolist()
            
            # Extract thinking and content
            thinking, content = extract_thinking_and_content(tokenizer, response_ids)
            full_model_output = f"{thinking}\n{content}" if thinking else content
            
            # Add to conversation
            episode_results[episode_idx]['conversation_dialogue'].append({"role": "assistant", "content": full_model_output})
            
            # Process command
            result = task.process_single_command(content, ground_truths[episode_idx], episode_states[episode_idx])
            
            # Store turn information
            episode_results[episode_idx]['episode_commands'].append(content)
            episode_results[episode_idx]['episode_outputs'].append(result['command_output'])
            episode_results[episode_idx]['episode_rewards'].append(result['reward'])
            episode_results[episode_idx]['episode_thinking_contents'].append(thinking)
            episode_results[episode_idx]['episode_contents'].append(content)
            episode_results[episode_idx]['turn_count'] = len(episode_results[episode_idx]['episode_commands'])
            
            # Update episode state
            episode_states[episode_idx] = task.update_episode_state(episode_states[episode_idx], result['command_output'])
            episode_states[episode_idx]['terminal_context'] = result['terminal_context']
            episode_states[episode_idx]['terminal_env'] = task.terminal_env
            
            # Handle no command case
            if not content.strip():
                episode_results[episode_idx]['conversation_dialogue'].append({"role": "human", "content": NO_COMMAND_MESSAGE})
                continue
            
            # Process terminal results and add human message
            if content.strip():
                if result.get('reason') == 'no_command':
                    episode_results[episode_idx]['conversation_dialogue'].append({"role": "human", "content": NO_COMMAND_MESSAGE})
                elif result.get('terminal_context', '') and result.get('reason') != 'no_command':
                    terminal_context = result.get('terminal_context', '')
                    terminal_context = episode_states[episode_idx]['terminal_env'].get_context()
                    
                    if result.get('reward', 0.0) == 0.0 and result.get('is_verifier', False):
                        terminal_message = get_verifier_incorrect_message(terminal_context)
                        episode_results[episode_idx]['conversation_dialogue'].append({"role": "human", "content": terminal_message})
                    elif result.get('reward', 0.0) == 1.0 and result.get('is_verifier', False):
                        # Episode completed successfully
                        episode_results[episode_idx]['episode_complete'] = True
                        episode_results[episode_idx]['final_reward'] = result['reward']
                        completed_episodes.add(episode_idx)
                    else:
                        terminal_message = get_normal_terminal_message(terminal_context)
                        episode_results[episode_idx]['conversation_dialogue'].append({"role": "human", "content": terminal_message})
            
            # Check if episode complete
            if result['episode_complete']:
                episode_results[episode_idx]['episode_complete'] = True
                episode_results[episode_idx]['final_reward'] = task.get_episode_reward(episode_states[episode_idx].get('terminal_env')) if result['episode_complete'] else 0.0
                completed_episodes.add(episode_idx)
        
        # Remove completed episodes from active list
        for episode_idx in completed_episodes:
            if episode_idx in active_episodes:
                active_episodes.remove(episode_idx)

    # Set final rewards for episodes that didn't complete
    for episode_idx in range(batch_size):
        if not episode_results[episode_idx]['episode_complete']:
            episode_results[episode_idx]['final_reward'] = task.get_episode_reward(episode_states[episode_idx].get('terminal_env'))
            episode_results[episode_idx]['terminal_context'] = episode_states[episode_idx].get('terminal_context', '')
    
    return episode_results


def apply_wrong_answer(batch: List[Dict], config: Config, a: int = 0, b: int = 1) -> None:
    """Corrupt ground-truth answers by adding a random non-zero integer in [a, b]

    The function mutates the batch **in-place** so callers can keep using the
    same list object. If the flag is off, the batch is left untouched.

    For every answer we
    1. Ensure it can be parsed as an integer (raise immediately otherwise).
    2. Sample a random offset `delta` from [a, b] \ {0} such that
       `answer + delta >= 0`.
    3. Apply the offset while preserving the original type (int vs. str).
    """
    if not getattr(config, 'wrong_answer', False):
        return  # No-op when feature disabled

    for idx, item in enumerate(batch):
        if 'answer' not in item:
            raise ValueError(f"wrong_answer enabled but batch item at index {idx} has no 'answer' field.")

        original_answer = item['answer']
        try:
            int_val = int(original_answer)
        except (ValueError, TypeError):
            raise ValueError(
                f"wrong_answer is True but answer '{original_answer}' at index {idx} is not an integer and cannot be perturbed."
            )

        # Sample a non-zero delta in the range [a, b] such that int_val + delta >= 0
        delta = 0
        while delta == 0 or int_val + delta < 0:
            delta = random.randint(a, b)

        new_val = int_val + delta

        # Preserve original type (e.g. keep string if it was a string)
        if isinstance(original_answer, str) and not isinstance(original_answer, int):
            item['answer'] = str(new_val)
        else:
            item['answer'] = new_val

# ============================================================================
# MAIN TRAINING FUNCTIONS
# ============================================================================

def _batch_judge_penalties(judge_penalty, prompts, contents, dialogues):
    """Helper to run judge_penalty.calculate_penalty_with_score in parallel for a batch."""
    async def _run_all():
        coros = [judge_penalty.calculate_penalty_with_score(prompts[i], contents[i], dialogues[i]) for i in range(len(prompts))]
        return await asyncio.gather(*coros)
    return asyncio.run(_run_all())

def _batch_judge_penalties_cot(judge_penalty, prompts, cot_contents, cot_dialogues):
    """Helper to run judge_penalty.calculate_penalty_cot_with_score in parallel for a batch."""
    async def _run_all():
        coros = [judge_penalty.calculate_penalty_cot_with_score(prompts[i], cot_contents[i], cot_dialogues[i]) for i in range(len(prompts))]
        return await asyncio.gather(*coros)
    return asyncio.run(_run_all())

def train_multi_turn(config_path: str = "config.yaml") -> None:
    """Multi-turn training function for terminal-based tasks with batching support."""
    config = load_config(config_path)

    # Setup
    face_model, tokenizer, device, shoggoth_model = load_model_and_tokenizer(config)
    task, wandb_logger, rollout_logger = setup_task_and_logging(config)
    optimizer, logit_processor = setup_training_components(config, face_model, tokenizer, shoggoth_model)
    custom_prompt = load_custom_prompt(config)
    judge_penalty = JudgePenalty(config)
    regex_penalty = RegexPenalty(config)

    # Set random seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Gradient accumulation variables
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    batch_size = config.batch_size
    accumulated_loss = 0.0
    accumulated_rewards = []          # total reward (after all penalties)
    accumulated_task_rewards = []    # reward from completing the task correctly (before judge penalty)
    accumulated_judge_penalties = [] # penalty from judge
    accumulated_regex_penalties = [] # penalty from regex
    accumulated_judge_penalties_cot = [] # penalty from judge on CoT (for logging only)
    accumulated_regex_penalties_cot = [] # penalty from regex on CoT (for logging only)
    # Add new accumulators for raw (unscaled) metrics
    accumulated_judge_scores = []  # raw judge scores (output)
    accumulated_judge_scores_cot = []  # raw judge scores (cot)
    accumulated_regex_word_counts = []  # raw regex word counts (output)
    accumulated_regex_word_counts_cot = []  # raw regex word counts (cot)
    accumulated_episodes = []

    # Calculate number of batches needed
    num_batches = (config.num_episodes + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        face_model.eval()
        if shoggoth_model is not None:
            shoggoth_model.eval()
        logit_processor.reset()

        # Prepare batch and prompts
        batch_indices = [random.randrange(len(task)) for _ in range(batch_size)]
        batch = [task[idx] for idx in batch_indices]
        # Optionally perturb answers if wrong_answer flag is enabled
        apply_wrong_answer(batch, config, a=0, b=1)
        prompts = build_prompts(config, batch, custom_prompt)
        ground_truths = [b['answer'] for b in batch]

        # Run batched multi-turn episodes
        episode_results = run_batched_multi_turn_episodes(
            face_model, tokenizer, task, prompts, logit_processor, config, device, ground_truths, shoggoth_model
        )

        # Prepare lists for judge penalty batch processing
        judge_args = []
        judge_cot_args = []
        for episode_idx, episode_result in enumerate(episode_results):
            if len(episode_result['episode_contents']) > 0:
                judge_args.append((prompts[episode_idx], episode_result['episode_contents'][-1], episode_result['conversation_dialogue']))
            else:
                judge_args.append((prompts[episode_idx], '', episode_result['conversation_dialogue']))
            # For CoT: pass CoT string as response, and None for conversation_dialogue so judge evaluates only the CoT
            if len(episode_result['episode_thinking_contents']) > 0:
                judge_cot_args.append((prompts[episode_idx], episode_result['episode_thinking_contents'][-1], None))
            else:
                judge_cot_args.append((prompts[episode_idx], '', None))

        # Batch judge penalty for outputs
        if judge_penalty is not None:
            judge_penalty_results = _batch_judge_penalties(
                judge_penalty,
                [a[0] for a in judge_args],
                [a[1] for a in judge_args],
                [a[2] for a in judge_args],
            )
        else:
            judge_penalty_results = [(0.0, 0.0)] * len(episode_results)

        # Batch judge penalty for CoT (async, memory efficient)
        if judge_penalty is not None:
            # Prepare CoT contents for batch processing
            cot_prompts = []
            cot_contents = []
            cot_dialogues = []
            for episode_idx, episode_result in enumerate(episode_results):
                cot_prompts.append(prompts[episode_idx])
                if len(episode_result['episode_thinking_contents']) > 0:
                    cot_contents.append(episode_result['episode_thinking_contents'][-1])
                else:
                    cot_contents.append('')
                # Pass conversation_dialogue so CoT judge can see all turns
                cot_dialogues.append(episode_result['conversation_dialogue'])
            
            # Use the new batched CoT judge penalty calculation
            judge_penalty_cot_results = _batch_judge_penalties_cot(
                judge_penalty,
                cot_prompts,
                cot_contents,
                cot_dialogues
            )
        else:
            judge_penalty_cot_results = [(0.0, 0.0)] * len(episode_results)

        # Collect all episode rewards first to calculate batch-level advantage
        batch_rewards = []  # penalized reward (task + penalties)
        batch_task_rewards_list = []  # pure task rewards (base)
        episode_data_list = []
        
        for episode_idx, episode_result in enumerate(episode_results):
            # Prepare for training
            base_rewards = [episode_result['final_reward']]
            penalties = [0.0]
            thinking_penalties = [0.0]
            penalized_rewards = [episode_result['final_reward']]

            # Use batch judge penalty results (output only)
            judge_penalty_value, judge_score = judge_penalty_results[episode_idx]
            # Only apply output judge penalty if not disabled in config
            if not getattr(config, 'judge_output_penalty_disabled', False):
                penalized_rewards[0] -= judge_penalty_value

            # Apply regex penalty if enabled (only on final content, not CoT)
            regex_penalty_value = 0.0
            regex_word_count = 0  # raw count of forbidden words in final answer
            if regex_penalty is not None and len(episode_result['episode_contents']) > 0:
                regex_penalty_value, regex_word_penalties = regex_penalty.calculate_penalty(
                    prompts[episode_idx], episode_result['episode_contents'][-1],
                    episode_result['conversation_dialogue']
                )
                # Sum raw counts across all forbidden words
                if regex_word_penalties:
                    regex_word_count = sum(v['count'] for v in regex_word_penalties.values())
                penalized_rewards[0] -= regex_penalty_value

            # Use batch judge penalty results for CoT (logging only, never used in reward)
            judge_penalty_cot, judge_score_cot = judge_penalty_cot_results[episode_idx]
            regex_penalty_cot = 0.0
            regex_word_count_cot = 0
            if len(episode_result['episode_thinking_contents']) > 0 and regex_penalty is not None:
                regex_penalty_cot, regex_word_penalties_cot = regex_penalty.calculate_penalty_cot_only(
                    prompts[episode_idx], episode_result['episode_thinking_contents'][-1],
                    episode_result['conversation_dialogue']
                )
                if regex_word_penalties_cot:
                    regex_word_count_cot = sum(v['count'] for v in regex_word_penalties_cot.values())

            # Extract what the CoT judge sees for logging
            cot_judge_content = ""
            if judge_penalty is not None and episode_result['conversation_dialogue']:
                cot_judge_content = judge_penalty._extract_cot_dialogue(episode_result['conversation_dialogue'])

            # Store episode data for later processing
            episode_data = {
                'episode_idx': episode_idx,
                'episode_result': episode_result,
                'base_rewards': base_rewards,
                'penalties': penalties,
                'thinking_penalties': thinking_penalties,
                'penalized_rewards': penalized_rewards,  # Only output penalty used
                'judge_penalty_value': judge_penalty_value,
                'judge_score': judge_score,
                'regex_penalty_value': regex_penalty_value,
                'regex_word_count': regex_word_count,
                'judge_penalty_cot': judge_penalty_cot,  # Only for logging
                'judge_score_cot': judge_score_cot,
                'regex_penalty_cot': regex_penalty_cot,
                'regex_word_count_cot': regex_word_count_cot,
                'cot_judge_content': cot_judge_content,  # What the CoT judge sees
                'prompt': prompts[episode_idx],
                'batch_item': batch[episode_idx]
            }
            episode_data_list.append(episode_data)
            batch_rewards.extend(penalized_rewards)
            batch_task_rewards_list.extend(base_rewards)
        
        # Calculate batch-level advantage
        batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        batch_task_rewards_tensor = torch.tensor(batch_task_rewards_list, dtype=torch.float32, device=device)

        batch_baseline_total = batch_rewards_tensor.mean().detach()
        batch_baseline_task  = batch_task_rewards_tensor.mean().detach()

        batch_advantages_total = batch_rewards_tensor - batch_baseline_total
        batch_advantages_task  = batch_task_rewards_tensor - batch_baseline_task

        # Now process each episode with the correct advantage
        for episode_data in episode_data_list:
            episode_idx = episode_data['episode_idx']
            episode_result = episode_data['episode_result']
            
            # Convert to tensors
            base_rewards = torch.tensor(episode_data['base_rewards'], dtype=torch.float32, device=device)
            penalties = torch.tensor(episode_data['penalties'], dtype=torch.float32, device=device)
            thinking_penalties = torch.tensor(episode_data['thinking_penalties'], dtype=torch.float32, device=device)
            rewards = torch.tensor(episode_data['penalized_rewards'], dtype=torch.float32, device=device)
            
            # Get the advantages for this episode
            total_advantage_ep = batch_advantages_total[episode_idx:episode_idx+1]
            task_advantage_ep  = batch_advantages_task[episode_idx:episode_idx+1]

            # Training step - train on all turns of the episode
            if ((batch_idx * batch_size + episode_idx) % gradient_accumulation_steps) == 0:
                optimizer.zero_grad()

            # Determine which turns to backprop through. Training on assistant tokens only
            total_loss = 0.0
            num_turns = len(episode_result['turn_input_ids'])

            selected_turn_indices = range(num_turns)

            for turn_idx in selected_turn_indices:
                # Get training data for this turn
                target_tokens = episode_result['turn_target_tokens'][turn_idx]
                mask = episode_result['turn_masks'][turn_idx]
                input_ids = episode_result['turn_input_ids'][turn_idx]
                
                if shoggoth_model is None:
                    # Single-model path ---------------------------------------------------
                    logits = face_model(input_ids).logits
                    turn_loss = perform_training_step(
                        face_model, optimizer, total_advantage_ep, task_advantage_ep, logits, target_tokens, mask, episode_result['turn_think_masks'][turn_idx].to(device), config, gradient_accumulation_steps, input_ids
                    )
                else:
                    # Dual-model path -----------------------------------------------------
                    think_mask = episode_result['turn_think_masks'][turn_idx].to(device)
                    turn_loss = perform_training_step_dual(
                        face_model,
                        shoggoth_model,
                        optimizer,
                        total_advantage_ep,
                        task_advantage_ep,
                        target_tokens,
                        mask,
                        think_mask,
                        gradient_accumulation_steps,
                        input_ids,
                        config
                    )
                
                total_loss += turn_loss
            
            # Average the loss across the turns we actually trained on
            loss = total_loss / (len(selected_turn_indices) if selected_turn_indices else 1)

            # Store metrics
            accumulated_loss += loss
            accumulated_rewards.extend(rewards.tolist())
            accumulated_task_rewards.extend(base_rewards.tolist())
            accumulated_judge_penalties.append(episode_data['judge_penalty_value'])
            accumulated_regex_penalties.append(episode_data['regex_penalty_value'])
            accumulated_judge_penalties_cot.append(episode_data['judge_penalty_cot'])
            accumulated_regex_penalties_cot.append(episode_data['regex_penalty_cot'])
            # Add raw metrics to accumulation
            accumulated_judge_scores.append(episode_data['judge_score'])
            accumulated_judge_scores_cot.append(episode_data['judge_score_cot'])
            accumulated_regex_word_counts.append(episode_data['regex_word_count'])
            accumulated_regex_word_counts_cot.append(episode_data['regex_word_count_cot'])
            accumulated_episodes.append({
                'episode': batch_idx * batch_size + episode_idx,
                'prompts': [prompts[episode_idx]],
                'targets': [batch[episode_idx]["answer"]],
                'thinking_contents': episode_result['episode_thinking_contents'],
                'contents': episode_result['episode_contents'],
                'rewards': rewards.tolist(),
                'base_rewards': base_rewards.tolist(),
                'penalties': penalties.tolist(),
                'thinking_penalties': thinking_penalties.tolist(),
                'judge_scores': [episode_data['judge_score']],
                'judge_penalty_cot': episode_data['judge_penalty_cot'],
                'judge_score_cot': episode_data['judge_score_cot'],
                'regex_penalty_cot': episode_data['regex_penalty_cot'],
                'regex_word_count_cot': episode_data['regex_word_count_cot'],
                'cot_judge_content': episode_data['cot_judge_content'],  # What the CoT judge sees
                'loss': loss,
                'turn_count': episode_result['turn_count'],
                'episode_complete': episode_result['episode_complete'],
                'final_reward': episode_result['final_reward'],
                'commands': episode_result['episode_commands'],
                'command_outputs': episode_result['episode_outputs'],
                'terminal_context': episode_result['terminal_context'],
                'episode_rewards': episode_result['episode_rewards'],
                'conversation_dialogue': episode_result['conversation_dialogue'],
                'regex_penalties': [episode_data['regex_penalty_value']],
                'regex_word_counts': [episode_data['regex_word_count']],
            })

        # Optimizer step and logging
        if ((batch_idx + 1) * batch_size) % gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
            if shoggoth_model is None:
                torch.nn.utils.clip_grad_norm_(face_model.parameters(), 1.0)
                zero_special_token_grads(face_model, tokenizer)
            else:
                all_params = list(face_model.parameters()) + list(shoggoth_model.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                zero_special_token_grads(face_model, tokenizer)
                zero_special_token_grads(shoggoth_model, tokenizer)

            optimizer.step()

            # Log metrics
            avg_loss = accumulated_loss / len(accumulated_episodes) if accumulated_episodes else 0.0
            avg_total_reward = sum(accumulated_rewards) / len(accumulated_rewards) if accumulated_rewards else 0.0
            avg_task_reward  = sum(accumulated_task_rewards) / len(accumulated_task_rewards) if accumulated_task_rewards else 0.0
            avg_judge_penalty = sum(accumulated_judge_penalties) / len(accumulated_judge_penalties) if accumulated_judge_penalties else 0.0
            avg_regex_penalty = sum(accumulated_regex_penalties) / len(accumulated_regex_penalties) if accumulated_regex_penalties else 0.0
            avg_judge_penalty_cot = sum(accumulated_judge_penalties_cot) / len(accumulated_judge_penalties_cot) if accumulated_judge_penalties_cot else 0.0
            avg_regex_penalty_cot = sum(accumulated_regex_penalties_cot) / len(accumulated_regex_penalties_cot) if accumulated_regex_penalties_cot else 0.0
            avg_turn_count = sum(ep['turn_count'] for ep in accumulated_episodes) / len(accumulated_episodes) if accumulated_episodes else 0.0

            # Raw (unscaled) metrics averages
            avg_judge_score_output = sum(accumulated_judge_scores) / len(accumulated_judge_scores) if accumulated_judge_scores else 0.0
            avg_judge_score_cot = sum(accumulated_judge_scores_cot) / len(accumulated_judge_scores_cot) if accumulated_judge_scores_cot else 0.0
            avg_word_count_output = sum(accumulated_regex_word_counts) / len(accumulated_regex_word_counts) if accumulated_regex_word_counts else 0.0
            avg_word_count_cot = sum(accumulated_regex_word_counts_cot) / len(accumulated_regex_word_counts_cot) if accumulated_regex_word_counts_cot else 0.0
            completion_rate = sum(1 for ep in accumulated_episodes if ep['episode_complete']) / len(accumulated_episodes) if accumulated_episodes else 0.0

            # Compute task reward means conditioned on judge score when judge is enabled
            task_rewards_high_judge, task_rewards_low_judge = [], []
            if judge_penalty is not None and getattr(judge_penalty, 'enabled', False):
                for ep in accumulated_episodes:
                    judge_scores_ep = ep.get('judge_scores', [])
                    base_rewards_ep = ep.get('base_rewards', [])
                    for js, br in zip(judge_scores_ep, base_rewards_ep):
                        if js > 0.5:
                            task_rewards_high_judge.append(br)
                        else:
                            task_rewards_low_judge.append(br)
            mean_task_reward_high_judge = (sum(task_rewards_high_judge) / len(task_rewards_high_judge)) if task_rewards_high_judge else float('nan')
            mean_task_reward_low_judge  = (sum(task_rewards_low_judge)  / len(task_rewards_low_judge))  if task_rewards_low_judge  else float('nan')

            # Compute task reward means conditioned on regex penalty
            task_rewards_regex_zero, task_rewards_regex_nonzero = [], []
            if regex_penalty is not None and getattr(regex_penalty, 'enabled', False):
                for ep in accumulated_episodes:
                    regex_word_counts_ep = ep.get('regex_word_counts', [])
                    base_rewards_ep = ep.get('base_rewards', [])
                    for wc, br in zip(regex_word_counts_ep, base_rewards_ep):
                        if wc == 0:
                            task_rewards_regex_zero.append(br)
                        else:
                            task_rewards_regex_nonzero.append(br)
            mean_task_reward_regex_zero = (sum(task_rewards_regex_zero) / len(task_rewards_regex_zero)) if task_rewards_regex_zero else float('nan')
            mean_task_reward_regex_nonzero = (sum(task_rewards_regex_nonzero) / len(task_rewards_regex_nonzero)) if task_rewards_regex_nonzero else float('nan')

            wandb_logger.log({
                    "loss": avg_loss,
                    "total_reward_mean": avg_total_reward,
                    "task_reward_mean":  avg_task_reward,
                    "judge_penalty_mean": avg_judge_penalty,
                    "regex_penalty_mean": avg_regex_penalty,
                    "judge_penalty_mean_cot": avg_judge_penalty_cot,
                    "regex_penalty_mean_cot": avg_regex_penalty_cot,
                    "avg_turn_count": avg_turn_count,
                    "completion_rate": completion_rate,
                    "judge_score_output": avg_judge_score_output,
                    "judge_score_cot": avg_judge_score_cot,
                    "word_count_output": avg_word_count_output,
                    "word_count_cot": avg_word_count_cot,
                    "task_reward_mean_judge_high": mean_task_reward_high_judge,
                    "task_reward_mean_judge_low": mean_task_reward_low_judge,
                    "task_reward_mean_regex_zero": mean_task_reward_regex_zero,
                    "task_reward_mean_regex_nonzero": mean_task_reward_regex_nonzero,
                    "max_turns": getattr(config, 'max_turns', 10),
                    "judge_output_penalty_disabled": getattr(config, 'judge_output_penalty_disabled', False),
            }, step=batch_idx)

            # Log rollouts in multiple formats
            for episode_data in accumulated_episodes:                
                # Also log in super readable text format
                rollout_logger.log_rollout(
                    episode=episode_data['episode'],
                    prompts=episode_data['prompts'],
                    targets=episode_data['targets'],
                    thinking_contents=episode_data['thinking_contents'],
                    contents=episode_data['contents'],
                    rewards=episode_data['rewards'],
                    loss=episode_data['loss'],
                    base_rewards=episode_data['base_rewards'],
                    thinking_penalties=episode_data['thinking_penalties'],
                    output_word_penalties=[{}],
                    thinking_word_penalties=[{}],
                    output_word_counts=[{}],
                    thinking_word_counts=[{}],
                    judge_scores=episode_data.get('judge_scores', []),
                    regex_penalties=episode_data.get('regex_penalties', []),
                    cot_judge_content=episode_data.get('cot_judge_content', ''),  # What the CoT judge sees
                    cot_judge_score=episode_data.get('judge_score_cot', None),  # Score given by CoT judge
                    turn_count=episode_data['turn_count'],
                    episode_complete=episode_data['episode_complete'],
                    final_reward=episode_data['final_reward'],
                    commands=episode_data['commands'],
                    command_outputs=episode_data['command_outputs'],
                    terminal_context=episode_data['terminal_context'],
                    episode_rewards=episode_data['episode_rewards'],
                    conversation_dialogue=episode_data['conversation_dialogue'],
                    format="super_readable"
                )

            # Reset accumulation variables
            accumulated_loss = 0.0
            accumulated_rewards = []
            accumulated_task_rewards = []
            accumulated_judge_penalties = []
            accumulated_regex_penalties = []
            accumulated_judge_penalties_cot = []
            accumulated_regex_penalties_cot = []
            # Reset raw metric accumulators
            accumulated_judge_scores = []
            accumulated_judge_scores_cot = []
            accumulated_regex_word_counts = []
            accumulated_regex_word_counts_cot = []
            accumulated_episodes = []

    # Conditionally save rollouts to W&B
    if getattr(config, 'save_rollouts_to_wandb', False) and wandb_logger.run is not None:
        rollouts_dir = "rollouts"
        txt_files = [f for f in os.listdir(rollouts_dir) if f.endswith("_super_readable.txt")]
        if txt_files:  # only create artifact if there is something to save
            combined_path = os.path.join(rollouts_dir, "super_readable_rollouts.jsonl")
            with open(combined_path, "w", encoding="utf-8") as fout:
                for filename in sorted(txt_files):
                    filepath = os.path.join(rollouts_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as fin:
                        content = fin.read()
                    json.dump({"filename": filename, "content": content}, fout)
                    fout.write("\n")
            wandb_logger.run.save(combined_path)

    wandb_logger.finish()