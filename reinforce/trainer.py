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
from .judge import JudgePenalty
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
                                  thinking_contents: List[str], judge_penalty: JudgePenalty = None) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Calculate rewards and penalties for the batch."""
    base_rewards, penalties, thinking_penalties, penalized_rewards, judge_penalties, judge_scores = [], [], [], [], [], []

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

        base_rewards.append(base_reward)
        penalties.append(penalty)
        thinking_penalties.append(thinking_penalty)
        judge_penalties.append(judge_penalty_value)
        judge_scores.append(judge_score)
        penalized_rewards.append(penalized_reward)

    return base_rewards, penalties, thinking_penalties, penalized_rewards, judge_penalties, judge_scores

def perform_training_step(model: Any, optimizer: Any, rewards: torch.Tensor,
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
    kl_penalty = compute_kl_penalty(
        rewards_tensor=rewards,
        ref_model=ref_model,
        config=config,
        full_ids=input_ids,
        response_start_idx=0,  # Will be calculated from input_ids
        response_logits=logits[:, :-1],
        gradient_mask=mask
    )

    baseline = rewards.mean().detach()
    advantage = rewards - baseline

    # Compute loss
    loss = -((advantage - config.kl_coefficient * kl_penalty.detach()) * logp_per_seq).mean() / gradient_accumulation_steps
    loss.backward()

    return loss.item() * gradient_accumulation_steps, kl_penalty.mean().item()

def run_multi_turn_episode(model: Any, tokenizer: Any, task: Any, initial_prompt: str,
                          logit_processor: Any, config: Config, device: str, ground_truth: Any) -> Dict[str, Any]:
    """Run a single multi-turn episode."""
    episode_state = task.create_episode_state()
    episode_rewards, episode_commands, episode_outputs = [], [], []
    episode_thinking_contents, episode_contents = [], []
    conversation_dialogue = [{"role": "human", "content": initial_prompt}]
    max_turns = getattr(config, 'max_turns', 10)
    for turn in range(max_turns):
        logit_processor.reset()

        # Build conversation for this turn
        if turn == 0:
            current_prompt = initial_prompt
            prompt_input = prepare_thinking_input(tokenizer, current_prompt, enable_thinking=True)
        else:
            # Manually construct the conversation format
            conversation_text = ""
            for msg in conversation_dialogue:
                if msg['role'] == 'human':
                    conversation_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg['role'] == 'assistant':
                    conversation_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            
            # Add the generation prompt
            conversation_text += "<|im_start|>assistant\n"
            prompt_input = conversation_text

        model_input = tokenizer([prompt_input], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **model_input,
                max_new_tokens=config.max_new_tokens,
                logits_processor=[logit_processor],
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = outputs.sequences
        prompt_len = model_input.input_ids.size(1)
        response_ids = generated_ids[0][prompt_len:].tolist()

        # Extract thinking and content
        thinking, content = extract_thinking_and_content(tokenizer, response_ids)
        full_model_output = f"{thinking}\n{content}" if thinking else content

        # Add to conversation
        conversation_dialogue.append({"role": "assistant", "content": full_model_output})

        # Process command
        result = task.process_single_command(content, ground_truth, episode_state)

        # Store turn information
        episode_commands.append(content)
        episode_outputs.append(result['command_output'])
        episode_rewards.append(result['reward'])
        episode_thinking_contents.append(thinking)
        episode_contents.append(content)

        # Update episode state
        episode_state = task.update_episode_state(episode_state, result['command_output'])
        episode_state['terminal_context'] = result['terminal_context']
        episode_state['terminal_env'] = task.terminal_env

        # Handle no command case
        if not content.strip():
            conversation_dialogue.append({"role": "human", "content": NO_COMMAND_MESSAGE})
            continue

        # Process terminal results and add human message
        if content.strip():
            if result.get('reason') == 'no_command':
                conversation_dialogue.append({"role": "human", "content": NO_COMMAND_MESSAGE})
            elif result.get('terminal_context', '') and result.get('reason') != 'no_command':
                terminal_context = result.get('terminal_context', '')
                terminal_context = episode_state['terminal_env'].get_context()

                if result.get('reward', 0.0) == 0.0 and result.get('is_verifier', False):
                    terminal_message = get_verifier_incorrect_message(terminal_context)
                    conversation_dialogue.append({"role": "human", "content": terminal_message})
                elif result.get('reward', 0.0) == 1.0 and result.get('is_verifier', False):
                    break
                else:
                    terminal_message = get_normal_terminal_message(terminal_context)
                    conversation_dialogue.append({"role": "human", "content": terminal_message})

        # Check if episode complete
        if result['episode_complete']:
            break

    # Determine final reward
    final_reward = task.get_episode_reward(episode_state.get('terminal_env')) if result['episode_complete'] else 0.0

    return {
        'episode_rewards': episode_rewards,
        'episode_commands': episode_commands,
        'episode_outputs': episode_outputs,
        'episode_thinking_contents': episode_thinking_contents,
        'episode_contents': episode_contents,
        'conversation_dialogue': conversation_dialogue,
        'final_reward': final_reward,
        'episode_complete': result['episode_complete'],
        'turn_count': len(episode_commands),
        'terminal_context': episode_state.get('terminal_context', '')
    }

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
        'terminal_context': ''
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
        
        with torch.no_grad():
            outputs = model.generate(
                **model_input,
                max_new_tokens=config.max_new_tokens,
                logits_processor=[logit_processor],
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        generated_ids = outputs.sequences
        prompt_lens = model_input.attention_mask.sum(dim=1)
        
        # Process each active episode
        completed_episodes = set()
        
        for i, episode_idx in enumerate(active_indices):
            prompt_len = prompt_lens[i].item()
            response_ids = generated_ids[i][prompt_len:].tolist()
            
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

    # Set random seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Gradient accumulation variables
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    batch_size = config.batch_size
    accumulated_loss = 0.0
    accumulated_rewards = []
    accumulated_judge_penalties = []
    accumulated_episodes = []

    # Calculate number of batches needed
    num_batches = (config.num_episodes + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        model.eval()
        logit_processor.reset()

        # Prepare batch and prompts
        batch_indices = [random.randrange(len(task)) for _ in range(batch_size)]
        batch = [task[idx] for idx in batch_indices]
        prompts = build_prompts(config, batch, custom_prompt)
        ground_truths = [b['answer'] for b in batch]

        # Run batched multi-turn episodes
        episode_results = run_batched_multi_turn_episodes(
            model, tokenizer, task, prompts, logit_processor, config, device, ground_truths
        )

        # Process each episode in the batch
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

            # Convert to tensors
            base_rewards = torch.tensor(base_rewards, dtype=torch.float32, device=device)
            penalties = torch.tensor(penalties, dtype=torch.float32, device=device)
            thinking_penalties = torch.tensor(thinking_penalties, dtype=torch.float32, device=device)
            rewards = torch.tensor(penalized_rewards, dtype=torch.float32, device=device)

            # Training step
            if ((batch_idx * batch_size + episode_idx) % gradient_accumulation_steps) == 0:
                optimizer.zero_grad()

            # Get the last command's logits for training
            last_prompt_input = prepare_thinking_input(tokenizer, prompts[episode_idx], enable_thinking=True)
            last_model_input = tokenizer([last_prompt_input], return_tensors="pt").to(device)

            logits = model(last_model_input.input_ids).logits
            target_tokens = last_model_input.input_ids[:, 1:]

            # Build mask for response tokens
            seq_len = last_model_input.input_ids.size(1) - 1
            arange = torch.arange(seq_len, device=device).unsqueeze(0)
            start_idx = torch.tensor([last_model_input.input_ids.size(1) - 1], device=device).unsqueeze(1)
            mask = arange >= start_idx

            loss, kl_penalty_mean = perform_training_step(
                model, optimizer, rewards, logits, target_tokens, mask, ref_model, config, gradient_accumulation_steps, last_model_input.input_ids
            )

            # Store metrics
            accumulated_loss += loss
            accumulated_rewards.extend(rewards.tolist())
            accumulated_judge_penalties.append(judge_penalty_value)
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
            })

        # Optimizer step and logging
        if ((batch_idx + 1) * batch_size) % gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            zero_special_token_grads(model, tokenizer)
            optimizer.step()

            # Log metrics
            avg_loss = accumulated_loss / len(accumulated_episodes) if accumulated_episodes else 0.0
            avg_reward = sum(accumulated_rewards) / len(accumulated_rewards) if accumulated_rewards else 0.0
            avg_judge_penalty = sum(accumulated_judge_penalties) / len(accumulated_judge_penalties) if accumulated_judge_penalties else 0.0
            avg_turn_count = sum(ep['turn_count'] for ep in accumulated_episodes) / len(accumulated_episodes) if accumulated_episodes else 0.0
            completion_rate = sum(1 for ep in accumulated_episodes if ep['episode_complete']) / len(accumulated_episodes) if accumulated_episodes else 0.0

            wandb_logger.log({
                    "loss": avg_loss,
                "reward_mean": avg_reward,
                    "judge_penalty_mean": avg_judge_penalty,
                    "avg_turn_count": avg_turn_count,
                    "completion_rate": completion_rate,
                "max_turns": getattr(config, 'max_turns', 10),
            }, step=batch_idx)

            # Log rollouts in multiple formats
            for episode_data in accumulated_episodes:
                # Log in JSON format
                rollout_logger.log_rollout(
                    episode=episode_data['episode'],
                    prompts=episode_data['prompts'],
                    targets=episode_data['targets'],
                    thinking_contents=episode_data['thinking_contents'],
                    contents=episode_data['contents'],
                    rewards=episode_data['rewards'],
                    loss=episode_data['loss'],
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
                    format="json"
                )
                
                # Also log in super readable text format
                rollout_logger.log_rollout(
                    episode=episode_data['episode'],
                    prompts=episode_data['prompts'],
                    targets=episode_data['targets'],
                    thinking_contents=episode_data['thinking_contents'],
                    contents=episode_data['contents'],
                    rewards=episode_data['rewards'],
                    loss=episode_data['loss'],
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
            accumulated_judge_penalties = []
            accumulated_episodes = []

    wandb_logger.finish()

def train(config_path: str = "config.yaml") -> None:
    """Main training function for single-turn tasks."""
    config = load_config(config_path)

    # Check if multi-turn mode is enabled
    if getattr(config, 'enable_multi_turn', False):
        print("Multi-turn mode enabled, using multi-turn training function")
        return train_multi_turn(config_path)

    # Setup
    model, tokenizer, device = load_model_and_tokenizer(config)
    ref_model = setup_reference_model(config)
    task, wandb_logger, rollout_logger = setup_task_and_logging(config)
    optimizer, logit_processor = setup_training_components(config, model, tokenizer)
    custom_prompt = load_custom_prompt(config)
    judge_penalty = JudgePenalty(config)

    # Set random seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Gradient accumulation variables
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    accumulated_loss = 0.0
    accumulated_rewards = []
    accumulated_base_rewards = []
    accumulated_penalties = []
    accumulated_thinking_penalties = []
    accumulated_judge_penalties = []
    accumulated_advantages = []
    accumulated_kl_penalties = []
    accumulated_episodes = []

    for episode in range(config.num_episodes):
        model.eval()
        logit_processor.reset()

        # Prepare batch and prompts
        batch_indices = [random.randrange(len(task)) for _ in range(config.batch_size)]
        batch = [task[idx] for idx in batch_indices]
        prompts = build_prompts(config, batch, custom_prompt)
        targets = [b["answer"] for b in batch]

        # Generate responses
        thinking_contents, contents = generate_responses(model, tokenizer, prompts, logit_processor, config, device)

        # Calculate rewards and penalties
        base_rewards, penalties, thinking_penalties, penalized_rewards, judge_penalties, judge_scores = calculate_rewards_and_penalties(
            task, batch, contents, thinking_contents, judge_penalty
        )

        # Convert to tensors
        base_rewards = torch.tensor(base_rewards, dtype=torch.float32, device=device)
        penalties = torch.tensor(penalties, dtype=torch.float32, device=device)
        thinking_penalties = torch.tensor(thinking_penalties, dtype=torch.float32, device=device)
        judge_penalties = torch.tensor(judge_penalties, dtype=torch.float32, device=device)
        rewards = torch.tensor(penalized_rewards, dtype=torch.float32, device=device)

        # Training step
        if (episode % gradient_accumulation_steps) == 0:
            optimizer.zero_grad()

        # Build full sequences for training
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

        # Build full sequences for training
        full_ids_list, start_indices = [], []
        for prompt_ids, seq in zip(model_inputs.input_ids, generated_ids):
            p_len = prompt_ids.size(0)
            start_indices.append(p_len - 1)
            full_ids_list.append(seq)
        full_ids = torch.nn.utils.rnn.pad_sequence(
            full_ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(device)

        # Forward pass
        logits = model(full_ids).logits
        target_tokens = full_ids[:, 1:]

        # Build mask
        seq_len = full_ids.size(1) - 1
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        start_idx = torch.tensor(start_indices, device=device).unsqueeze(1)
        mask = arange >= start_idx

        loss, kl_penalty_mean = perform_training_step(
            model, optimizer, rewards, logits, target_tokens, mask, ref_model, config, gradient_accumulation_steps, full_ids
        )

        # Store metrics
        accumulated_loss += loss
        accumulated_rewards.extend(rewards.tolist())
        accumulated_base_rewards.extend(base_rewards.tolist())
        accumulated_penalties.extend(penalties.tolist())
        accumulated_thinking_penalties.extend(thinking_penalties.tolist())
        accumulated_judge_penalties.extend(judge_penalties.tolist())
        accumulated_kl_penalties.extend([kl_penalty_mean] * len(rewards))

        episode_data = {
            'episode': episode,
            'prompts': prompts,
            'targets': targets,
            'thinking_contents': thinking_contents,
            'contents': contents,
            'rewards': rewards.tolist(),
            'base_rewards': base_rewards.tolist(),
            'penalties': penalties.tolist(),
            'thinking_penalties': thinking_penalties.tolist(),
            'judge_scores': judge_scores,
            'loss': loss,
            'kl_penalty_mean': kl_penalty_mean,
        }
        accumulated_episodes.append(episode_data)

        # Optimizer step and logging
        if (episode + 1) % gradient_accumulation_steps == 0 or episode == config.num_episodes - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            zero_special_token_grads(model, tokenizer)
            optimizer.step()

            # Log metrics
            avg_loss = accumulated_loss / len(accumulated_episodes) if accumulated_episodes else 0.0
            avg_reward = sum(accumulated_rewards) / len(accumulated_rewards) if accumulated_rewards else 0.0
            avg_base_reward = sum(accumulated_base_rewards) / len(accumulated_base_rewards) if accumulated_base_rewards else 0.0
            avg_penalty = sum(accumulated_penalties) / len(accumulated_penalties) if accumulated_penalties else 0.0
            avg_thinking_penalty = sum(accumulated_thinking_penalties) / len(accumulated_thinking_penalties) if accumulated_thinking_penalties else 0.0
            avg_judge_penalty = sum(accumulated_judge_penalties) / len(accumulated_judge_penalties) if accumulated_judge_penalties else 0.0
            avg_kl_penalty = sum(accumulated_kl_penalties) / len(accumulated_kl_penalties) if accumulated_kl_penalties else 0.0

            wandb_logger.log({
                    "loss": avg_loss,
                "reward_mean": avg_base_reward,
                "penalized_reward": avg_reward,
                "penalty": avg_penalty,
                    "thinking_penalty_mean": avg_thinking_penalty,
                    "judge_penalty_mean": avg_judge_penalty,
                    "kl_penalty_mean": avg_kl_penalty,
                    "kl_penalty_scaled": (config.kl_coefficient * avg_kl_penalty),
                    "gradient_accumulation_step": episode // gradient_accumulation_steps,
            }, step=episode)

            for episode_data in accumulated_episodes:
                rollout_logger.log_rollout(
                    episode=episode_data['episode'],
                    prompts=episode_data['prompts'],
                    targets=episode_data['targets'],
                    thinking_contents=episode_data['thinking_contents'],
                    contents=episode_data['contents'],
                    rewards=episode_data['rewards'],
                    loss=episode_data['loss'],
                    thinking_penalties=episode_data['thinking_penalties'],
                    output_word_penalties=[{}],
                    thinking_word_penalties=[{}],
                    output_word_counts=[{}],
                    thinking_word_counts=[{}],
                    kl_penalty_mean=episode_data['kl_penalty_mean'],
                    judge_scores=episode_data.get('judge_scores', []),
                    format="super_readable"
                )

            # Reset accumulation variables
            accumulated_loss = 0.0
            accumulated_rewards = []
            accumulated_base_rewards = []
            accumulated_penalties = []
            accumulated_thinking_penalties = []
            accumulated_judge_penalties = []
            accumulated_kl_penalties = []
            accumulated_episodes = []

    wandb_logger.finish()

    # Push to HuggingFace Hub if specified
    repo_out = getattr(config, 'hf_repo_out', None)
    if repo_out:
        try:
            print(f"Pushing model to HuggingFace Hub repo '{repo_out}' ...")
            model.cpu()
            model.push_to_hub(repo_out)
            try:
                tokenizer.push_to_hub(repo_out)
            except Exception as tok_err:
                print(f"Tokenizer push failed: {tok_err}")
            print("Successfully pushed to HuggingFace Hub.")
        except Exception as push_err:
            print(f"Failed to push model to HuggingFace Hub: {push_err}")