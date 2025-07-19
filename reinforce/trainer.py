import torch
import torch.nn.functional as F
import random
import yaml
import os
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from wandb_logger import WandbLogger
from .rollout_logger import RolloutLogger
from .kl_penalty import compute_kl_penalty
from .logit_processor import BatchThinkingTokenBudgetProcessor
from .utils import zero_special_token_grads
from .judge import JudgePenalty, RegexPenalty
from prompts.terminal_prompts import (
    get_initial_terminal_instructions,
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
        # Ensure the off-by-one flag always exists so downstream code can rely on it
        if 'answer_off_by_one' not in self.__dict__:
            self.answer_off_by_one = False
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

def load_model_and_tokenizer(config: Config) -> Tuple[Any, Any, str]:
    """Load model, tokenizer, and device."""
    model, tokenizer, device = load_qwen3_model(config.model_name, config.device)
    return model, tokenizer, device

def setup_reference_model(config: Config) -> Optional[Any]:
    """Setup reference model for KL penalty calculation."""
    if config.kl_coefficient <= 0:
        return None
    
    ref_model, _, _ = load_model_and_tokenizer(config)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    return ref_model

def setup_task_and_logging(config: Config) -> Tuple[Any, Any]:
    """Setup task and logging components."""
    task = load_task(config.task_name, config.custom_verifier_path, config)
    wandb_logger = WandbLogger(config)
    rollout_logger = RolloutLogger(config)
    return task, wandb_logger, rollout_logger

def setup_training_components(config: Config, model: Any, tokenizer: Any) -> Tuple[Any, Any]:
    """Setup optimizer and logit processor."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            enable_multi_turn = getattr(config, 'enable_multi_turn', False)
            custom_prompt = registry.get_prompt(config.task_name, use_terminal=use_terminal, enable_multi_turn=enable_multi_turn)
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
        enable_multi_turn = getattr(config, 'enable_multi_turn', False)
        prompts = [
            custom_prompt(b["question"], examples=None, metadata={
                "task_name": config.task_name,
                "use_terminal": use_terminal,
                "enable_multi_turn": enable_multi_turn
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
        enable_multi_turn = getattr(config, 'enable_multi_turn', False)
        if enable_multi_turn:
            prompts = [get_multi_turn_terminal_instructions(p) for p in prompts]
        else:
            prompts = [get_initial_terminal_instructions(p) for p in prompts]

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

def generate_responses(model: Any, tokenizer: Any, prompts: List[str],
                      logit_processor: Any, config: Config, device: str) -> Tuple[List[str], List[str]]:
    """Generate responses for the batch."""
    prompt_inputs = [prepare_thinking_input(tokenizer, p, enable_thinking=True) for p in prompts]
    model_inputs = tokenizer(prompt_inputs, return_tensors="pt", padding=True).to(device)
    prompt_lens = (model_inputs.input_ids != tokenizer.pad_token_id).sum(dim=1)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=config.max_new_tokens,
            logits_processor=[logit_processor],
            return_dict_in_generate=True,
            output_scores=True,
        )
    generated_ids = outputs.sequences

    thinking_contents, contents = [], []
    for seq, p_len in zip(generated_ids, prompt_lens):
        response_ids = seq[p_len:].tolist()
        thinking, content = extract_thinking_and_content(tokenizer, response_ids)
        thinking_contents.append(thinking)
        contents.append(content)

    return thinking_contents, contents

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

def perform_training_step(model: Any, optimizer: Any, advantage: torch.Tensor,
                        logits: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor,
                        ref_model: Any, config: Config, gradient_accumulation_steps: int, input_ids: torch.Tensor = None) -> Tuple[float, float]:
    """Perform the actual training step."""
    model.train()

    # Compute log-probs
    policy_log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    logp_taken = policy_log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
    
    # Average log-prob over response positions
    logp_per_seq = (logp_taken * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    # Compute KL penalty using centralized function
    # Note: We pass the advantage as rewards_tensor since KL penalty doesn't depend on the baseline
    kl_penalty = compute_kl_penalty(
        rewards_tensor=advantage,
        ref_model=ref_model,
        config=config,
        full_ids=input_ids,
        response_start_idx=0,  # Will be calculated from input_ids
        response_logits=logits[:, :-1],
        gradient_mask=mask
    )

    # Compute loss using the provided advantage
    loss = -(advantage * logp_per_seq).mean() / gradient_accumulation_steps
    loss.backward()

    return loss.item() * gradient_accumulation_steps, kl_penalty.mean().item()

def run_batched_multi_turn_episodes(model: Any, tokenizer: Any, task: Any, initial_prompts: List[str],
                                   logit_processor: Any, config: Config, device: str, ground_truths: List[Any]) -> List[Dict[str, Any]]:
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
        'turn_input_ids': []
    } for prompt in initial_prompts]
    
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

        # Generate responses for all active episodes
        model_input = tokenizer(active_prompts, return_tensors="pt", padding=True).to(device)
        
        # Generate the actual responses first
        with torch.no_grad():
            outputs = model.generate(
                **model_input,
                max_new_tokens=config.max_new_tokens,
                logits_processor=[logit_processor],
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        generated_ids = outputs.sequences
        prompt_lens = model_input.attention_mask.sum(dim=1)  # Actual prompt lengths for each sequence
                
        # Store training data for the full sequences (prompt + response)
        # We only store input_ids and compute logits fresh during training to ensure gradients
        seq_len = generated_ids.size(1) - 1
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        # For each sequence, the response starts at the actual prompt length
        # We want to mask all positions >= the start of the response
        response_start_positions = prompt_lens.unsqueeze(1)  # Shape: [batch_size, 1]
        mask = arange >= response_start_positions  # Shape: [batch_size, seq_len]
        
        # Store training data for each episode
        for i, episode_idx in enumerate(active_indices):
            episode_results[episode_idx]['turn_target_tokens'].append(generated_ids[i:i+1, 1:])
            episode_results[episode_idx]['turn_masks'].append(mask[i:i+1])
            episode_results[episode_idx]['turn_input_ids'].append(generated_ids[i:i+1])

        
        # Process each active episode
        completed_episodes = set()
        
        for i, episode_idx in enumerate(active_indices):
            response_ids = generated_ids[i][prompt_lens[i]:].tolist()
            
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

def apply_answer_off_by_one(batch: List[Dict], config: Config) -> None:
    """Increment ground-truth answers by +1 when configured.

    The function mutates the batch **in-place** so callers can keep using the
    same list object. If the flag is off, the batch is left untouched.

    An exception is raised immediately if any answer cannot be cast to an
    integer; this prevents silent training on corrupted data, as requested.
    """
    if not getattr(config, 'answer_off_by_one', False):
        return  # No-op when feature disabled

    for idx, item in enumerate(batch):
        if 'answer' not in item:
            raise ValueError(f"answer_off_by_one enabled but batch item at index {idx} has no 'answer' field.")
        original_answer = item['answer']
        try:
            int_val = int(original_answer)
        except (ValueError, TypeError):
            raise ValueError(
                f"answer_off_by_one is True but answer '{original_answer}' at index {idx} is not an integer and cannot be incremented."
            )

        incremented_val = int_val + 1
        # Preserve original type (e.g. keep string if it was a string)
        item['answer'] = str(incremented_val) if isinstance(original_answer, str) and not isinstance(original_answer, int) else incremented_val

# ============================================================================
# MAIN TRAINING FUNCTIONS
# ============================================================================

def train_multi_turn(config_path: str = "config.yaml") -> None:
    """Multi-turn training function for terminal-based tasks with batching support."""
    config = load_config(config_path)

    # Setup
    model, tokenizer, device = load_model_and_tokenizer(config)
    ref_model = setup_reference_model(config)
    task, wandb_logger, rollout_logger = setup_task_and_logging(config)
    optimizer, logit_processor = setup_training_components(config, model, tokenizer)
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
    accumulated_episodes = []

    # Calculate number of batches needed
    num_batches = (config.num_episodes + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        model.eval()
        logit_processor.reset()

        # Prepare batch and prompts
        batch_indices = [random.randrange(len(task)) for _ in range(batch_size)]
        batch = [task[idx] for idx in batch_indices]
        # Optionally shift integer answers by +1 according to configuration
        apply_answer_off_by_one(batch, config)
        prompts = build_prompts(config, batch, custom_prompt)
        ground_truths = [b['answer'] for b in batch]

        # Run batched multi-turn episodes
        episode_results = run_batched_multi_turn_episodes(
            model, tokenizer, task, prompts, logit_processor, config, device, ground_truths
        )

        # Collect all episode rewards first to calculate batch-level advantage
        batch_rewards = []
        episode_data_list = []
        
        for episode_idx, episode_result in enumerate(episode_results):
            # Prepare for training
            base_rewards = [episode_result['final_reward']]
            penalties = [0.0]
            thinking_penalties = [0.0]
            penalized_rewards = [episode_result['final_reward']]

            # Apply judge penalty if enabled
            judge_penalty_value = 0.0
            judge_score = 0.0
            if judge_penalty is not None and len(episode_result['episode_contents']) > 0:
                judge_penalty_value, judge_score = judge_penalty.calculate_penalty_sync_with_score(
                    prompts[episode_idx], episode_result['episode_contents'][-1], 
                    episode_result['conversation_dialogue']
                )
                penalized_rewards[0] -= judge_penalty_value

            # Apply regex penalty if enabled
            regex_penalty_value = 0.0
            if regex_penalty is not None and len(episode_result['episode_contents']) > 0:
                regex_penalty_value, _ = regex_penalty.calculate_penalty(
                    prompts[episode_idx], episode_result['episode_contents'][-1], 
                    episode_result['conversation_dialogue']
                )
                penalized_rewards[0] -= regex_penalty_value

            # Store episode data for later processing
            episode_data = {
                'episode_idx': episode_idx,
                'episode_result': episode_result,
                'base_rewards': base_rewards,
                'penalties': penalties,
                'thinking_penalties': thinking_penalties,
                'penalized_rewards': penalized_rewards,
                'judge_penalty_value': judge_penalty_value,
                'judge_score': judge_score,
                'regex_penalty_value': regex_penalty_value,
                'prompt': prompts[episode_idx],
                'batch_item': batch[episode_idx]
            }
            episode_data_list.append(episode_data)
            batch_rewards.extend(penalized_rewards)
        
        # Calculate batch-level advantage
        batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        batch_baseline = batch_rewards_tensor.mean().detach()
        batch_advantages = batch_rewards_tensor - batch_baseline
        
        # Now process each episode with the correct advantage
        for episode_data in episode_data_list:
            episode_idx = episode_data['episode_idx']
            episode_result = episode_data['episode_result']
            
            # Convert to tensors
            base_rewards = torch.tensor(episode_data['base_rewards'], dtype=torch.float32, device=device)
            penalties = torch.tensor(episode_data['penalties'], dtype=torch.float32, device=device)
            thinking_penalties = torch.tensor(episode_data['thinking_penalties'], dtype=torch.float32, device=device)
            rewards = torch.tensor(episode_data['penalized_rewards'], dtype=torch.float32, device=device)
            
            # Get the advantage for this episode
            episode_advantage = batch_advantages[episode_idx:episode_idx+1]

            # Training step - train on all turns of the episode
            if ((batch_idx * batch_size + episode_idx) % gradient_accumulation_steps) == 0:
                optimizer.zero_grad()

            # Train on each turn of the episode
            total_loss = 0.0
            total_kl_penalty = 0.0
            num_turns = len(episode_result['turn_input_ids'])
            
            for turn_idx in range(num_turns):
                # Get training data for this turn
                target_tokens = episode_result['turn_target_tokens'][turn_idx]
                mask = episode_result['turn_masks'][turn_idx]
                input_ids = episode_result['turn_input_ids'][turn_idx]
                
                # Compute logits fresh during training to ensure gradients
                logits = model(input_ids).logits
                
                # Use the batch-level advantage for all turns of this episode
                turn_loss, turn_kl_penalty = perform_training_step(
                    model, optimizer, episode_advantage, logits, target_tokens, mask, ref_model, config, gradient_accumulation_steps, input_ids
                )
                
                total_loss += turn_loss
                total_kl_penalty += turn_kl_penalty
            
            # Average the losses across turns
            loss = total_loss / num_turns if num_turns > 0 else 0.0
            kl_penalty_mean = total_kl_penalty / num_turns if num_turns > 0 else 0.0

            # Store metrics
            accumulated_loss += loss
            accumulated_rewards.extend(rewards.tolist())
            accumulated_task_rewards.extend(base_rewards.tolist())
            accumulated_judge_penalties.append(judge_penalty_value)
            accumulated_regex_penalties.append(regex_penalty_value)
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
                'judge_scores': [judge_score],
                'loss': loss,
                'kl_penalty_mean': kl_penalty_mean,
                'turn_count': episode_result['turn_count'],
                'episode_complete': episode_result['episode_complete'],
                'final_reward': episode_result['final_reward'],
                'commands': episode_result['episode_commands'],
                'command_outputs': episode_result['episode_outputs'],
                'terminal_context': episode_result['terminal_context'],
                'episode_rewards': episode_result['episode_rewards'],
                'conversation_dialogue': episode_result['conversation_dialogue'],
                'regex_penalties': [regex_penalty_value],
            })

        # Optimizer step and logging
        if ((batch_idx + 1) * batch_size) % gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            zero_special_token_grads(model, tokenizer)
            optimizer.step()

            # Log metrics
            avg_loss = accumulated_loss / len(accumulated_episodes) if accumulated_episodes else 0.0
            avg_total_reward = sum(accumulated_rewards) / len(accumulated_rewards) if accumulated_rewards else 0.0
            avg_task_reward  = sum(accumulated_task_rewards) / len(accumulated_task_rewards) if accumulated_task_rewards else 0.0
            avg_judge_penalty = sum(accumulated_judge_penalties) / len(accumulated_judge_penalties) if accumulated_judge_penalties else 0.0
            avg_regex_penalty = sum(accumulated_regex_penalties) / len(accumulated_regex_penalties) if accumulated_regex_penalties else 0.0
            avg_turn_count = sum(ep['turn_count'] for ep in accumulated_episodes) / len(accumulated_episodes) if accumulated_episodes else 0.0
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
            mean_task_reward_high_judge = (sum(task_rewards_high_judge) / len(task_rewards_high_judge)) if task_rewards_high_judge else 0.0
            mean_task_reward_low_judge  = (sum(task_rewards_low_judge)  / len(task_rewards_low_judge))  if task_rewards_low_judge  else 0.0

            # Compute task reward means conditioned on regex penalty
            task_rewards_regex_zero, task_rewards_regex_nonzero = [], []
            if regex_penalty is not None and getattr(regex_penalty, 'enabled', False):
                for ep in accumulated_episodes:
                    regex_penalties_ep = ep.get('regex_penalties', [])
                    base_rewards_ep = ep.get('base_rewards', [])
                    for rp, br in zip(regex_penalties_ep, base_rewards_ep):
                        if rp == 0.0:
                            task_rewards_regex_zero.append(br)
                        else:
                            task_rewards_regex_nonzero.append(br)
            mean_task_reward_regex_zero = (sum(task_rewards_regex_zero) / len(task_rewards_regex_zero)) if task_rewards_regex_zero else 0.0
            mean_task_reward_regex_nonzero = (sum(task_rewards_regex_nonzero) / len(task_rewards_regex_nonzero)) if task_rewards_regex_nonzero else 0.0

            wandb_logger.log({
                    "loss": avg_loss,
                    "total_reward_mean": avg_total_reward,
                    "task_reward_mean":  avg_task_reward,
                    "judge_penalty_mean": avg_judge_penalty,
                    "regex_penalty_mean": avg_regex_penalty,
                    "avg_turn_count": avg_turn_count,
                    "completion_rate": completion_rate,
                    "task_reward_mean_judge_high": mean_task_reward_high_judge,
                    "task_reward_mean_judge_low": mean_task_reward_low_judge,
                    "task_reward_mean_regex_zero": mean_task_reward_regex_zero,
                    "task_reward_mean_regex_nonzero": mean_task_reward_regex_nonzero,
                    "max_turns": getattr(config, 'max_turns', 10),
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
                    kl_penalty_mean=episode_data['kl_penalty_mean'],
                    judge_scores=episode_data.get('judge_scores', []),
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
            accumulated_episodes = []

    wandb_logger.finish()