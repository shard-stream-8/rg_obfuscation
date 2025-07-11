import os
import random
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from reinforce.logit_processor import BatchThinkingTokenBudgetProcessor
from reinforce.rollout_logger import RolloutLogger
from reinforce.utils import zero_special_token_grads
from tasks.task_loader import load_task
from prompts.registry import registry
from tasks.prompt_templates import create_custom_prompt
from models.qwen3 import load_qwen3_model, prepare_thinking_input
from wandb_logger import WandbLogger

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

def train_multi_turn(config_path: str = "config.yaml") -> None:
    """
    Multi-turn training function for terminal-based tasks.
    Handles episodic multi-turn interactions while maintaining REINFORCE structure.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)

    # Initialize wandb logger
    wandb_logger = WandbLogger(config)

    # Load model and tokenizer
    model, tokenizer, device = load_qwen3_model(config.model_name, config.device)

    # Reference network for KL penalty (frozen parameters)
    ref_model, _, _ = load_qwen3_model(config.model_name, config.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Load task
    task = load_task(config.task_name, config.custom_verifier_path, config)

    # Load custom prompt if specified
    custom_prompt = None
    if hasattr(config, 'custom_prompt_path') and config.custom_prompt_path:
        if config.custom_prompt_path == "registry":
            # Use the registry system
            try:
                from prompts.registry import registry
                # Get terminal configuration from config
                use_terminal = getattr(config, 'use_terminal', False)
                enable_multi_turn = getattr(config, 'enable_multi_turn', False)
                
                custom_prompt = registry.get_prompt(config.task_name, use_terminal=use_terminal, enable_multi_turn=enable_multi_turn)
                if custom_prompt is not None:
                    print(f"Loaded custom prompt for task '{config.task_name}' from registry")
                else:
                    print(f"No custom prompt found in registry for task '{config.task_name}', using default")
            except ImportError:
                print("Prompt registry not available, falling back to default prompt")
        else:
            # Use the old file-based approach
            custom_prompt = registry.load_prompt_from_file(config.custom_prompt_path)

    # Initialize components
    logit_processor = BatchThinkingTokenBudgetProcessor(
        tokenizer,
        max_thinking_tokens=config.max_thinking_tokens,
        batch_size=1,  # Single example for multi-turn
        min_thinking_tokens=config.min_thinking_tokens,
    )
    rollout_logger = RolloutLogger(config)

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=1e-2,
    )

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
    accumulated_advantages = []
    accumulated_kl_penalties = []
    accumulated_episodes = []

    max_turns = getattr(config, 'max_turns', 10)

    for episode in range(config.num_episodes):
        model.eval()

        # Reset logit processor state for new episode
        logit_processor.reset()

        # For multi-turn, we'll handle one example at a time for simplicity
        batch_indices = [random.randrange(len(task)) for _ in range(1)]  # Single example
        batch = [task[idx] for idx in batch_indices]
        
        # Apply custom prompt if available
        if custom_prompt is not None:
            print(f"DEBUG: Using custom prompt for task {config.task_name}")
            # Get terminal configuration for metadata
            use_terminal = getattr(config, 'use_terminal', False)
            enable_multi_turn = getattr(config, 'enable_multi_turn', False)
            
            prompts = [
                custom_prompt(b["question"], examples=None, metadata={
                    "task_name": config.task_name,
                    "use_terminal": use_terminal,
                    "enable_multi_turn": enable_multi_turn
                })
                if callable(custom_prompt) else custom_prompt(b["question"])
                for b in batch
            ]
        else:
            print(f"No custom prompt available, using default")
            prompts = [b["question"] for b in batch]
        
        # Prepend terminal instructions if needed
        if getattr(config, 'use_terminal', False):
            prompts = [prepend_terminal_instructions(p) for p in prompts]
            
        targets = [b["answer"] for b in batch]

        # Multi-turn interaction loop
        episode_state = task.create_episode_state()
        episode_rewards = []
        episode_commands = []
        episode_outputs = []
        episode_thinking_contents = []
        episode_contents = []
        episode_complete = False
        final_reward = 0.0
        
        # Store all model outputs and terminal contexts for context building
        all_model_outputs = []
        all_terminal_contexts = []

        # Store conversation history - build incrementally
        conversation_dialogue = [{"role": "human", "content": prompts[0]}]

        for turn in range(max_turns):
            # Build the conversation for this turn
            if turn == 0:
                # First turn - just the initial prompt
                current_prompt = prompts[0]
            else:
                # For subsequent turns, use the conversation_dialogue that we've been building
                # Apply the conversation template
                current_prompt = tokenizer.apply_chat_template(
                    conversation_dialogue,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            # Generate single command
            prompt_input = prepare_thinking_input(tokenizer, current_prompt, enable_thinking=True)
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
            
            # Decode response
            prompt_len = model_input.input_ids.size(1)
            response_ids = generated_ids[0][prompt_len:].tolist()
            
            # Extract thinking and content
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
                    thinking_ids = response_ids[thinking_start + 1:thinking_end - 1]
                    content_ids = response_ids[thinking_end:]
                    thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
                    content = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
                else:
                    thinking_ids = response_ids[thinking_start + 1:]
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
            
            # The full model output for this turn (including reasoning and commands)
            full_model_output = f"{thinking}\n{content}" if thinking else content
            all_model_outputs.append(full_model_output)
            
            # Add the assistant's response to the conversation dialogue
            conversation_dialogue.append({"role": "assistant", "content": full_model_output})
            
            # Process the single command
            result = task.process_single_command(content, batch[0]['answer'], episode_state)
            
            # Add the terminal output to the conversation dialogue
            if result.get('terminal_context', ''):
                terminal_message = f"Your command was executed. Here is the output:\n\n{result['terminal_context']}\n\nWhat's your next command?"
                conversation_dialogue.append({"role": "human", "content": terminal_message})
            
            # Store turn information
            episode_commands.append(content)
            episode_outputs.append(result['command_output'])
            episode_rewards.append(result['reward'])
            episode_thinking_contents.append(thinking)
            episode_contents.append(content)
            all_terminal_contexts.append(result.get('terminal_context', ''))
            
            # Update episode state
            episode_state = task.update_episode_state(episode_state, result['command_output'])
            episode_state['terminal_context'] = result['terminal_context']
            episode_state['terminal_env'] = task.terminal_env
            
            # If no command was generated, end the episode
            if not content.strip():
                episode_complete = True
                final_reward = 0.0
                break
            
            # Check if episode is complete (verifier returned 1 or max_turns reached)
            if result['episode_complete']:
                episode_complete = True
                final_reward = task.get_episode_reward(episode_state.get('terminal_env'))
                break
        
        # If episode didn't complete naturally, use final reward
        if not episode_complete:
            final_reward = 0.0  # No positive verifier result
        
        # Prepare for REINFORCE training
        # We'll use the final reward for the entire episode
        base_rewards = [final_reward]
        penalties = [0.0]  # No penalties in multi-turn for now
        thinking_penalties = [0.0]
        penalized_rewards = [final_reward]
        
        # Convert to tensors
        base_rewards = torch.tensor(base_rewards, dtype=torch.float32, device=device)
        penalties = torch.tensor(penalties, dtype=torch.float32, device=device)
        thinking_penalties = torch.tensor(thinking_penalties, dtype=torch.float32, device=device)
        rewards = torch.tensor(penalized_rewards, dtype=torch.float32, device=device)
        
        # For multi-turn, we need to handle the full episode sequence
        # This is a simplified version - in practice, you might want to use the full sequence
        # For now, we'll use the final command's logits for training
        
        # Get the last command's logits for training
        last_prompt_input = prepare_thinking_input(tokenizer, current_prompt, enable_thinking=True)
        last_model_input = tokenizer([last_prompt_input], return_tensors="pt").to(device)
        
        model.train()
        
        # Only zero gradients on the first accumulation step
        if (episode % gradient_accumulation_steps) == 0:
            optimizer.zero_grad()
        
        # Forward pass through the policy network
        logits = model(last_model_input.input_ids).logits
        
        # Compute log-probs for the last command
        policy_log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        target_tokens = last_model_input.input_ids[:, 1:]
        logp_taken = policy_log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Build mask for response tokens
        seq_len = last_model_input.input_ids.size(1) - 1
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        start_idx = torch.tensor([last_model_input.input_ids.size(1) - 1], device=device).unsqueeze(1)
        mask = arange >= start_idx
        
        # Average log-prob over response positions
        logp_per_seq = (logp_taken * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Compute KL penalty
        with torch.no_grad():
            ref_logits = ref_model(last_model_input.input_ids).logits[:, :-1]
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        kl_token = torch.nn.functional.kl_div(
            policy_log_probs,
            ref_log_probs.exp(),
            reduction="none",
        ).sum(-1)
        kl_penalty = (kl_token * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        baseline = rewards.mean().detach()
        advantage = rewards - baseline
        
        # Compute loss
        loss = -((advantage - config.kl_coefficient * kl_penalty.detach()) * logp_per_seq).mean() / gradient_accumulation_steps
        loss.backward()
        
        # Store metrics for logging
        accumulated_loss += loss.item() * gradient_accumulation_steps
        accumulated_rewards.extend(rewards.tolist())
        accumulated_base_rewards.extend(base_rewards.tolist())
        accumulated_penalties.extend(penalties.tolist())
        accumulated_thinking_penalties.extend(thinking_penalties.tolist())
        accumulated_advantages.extend(advantage.tolist())
        accumulated_kl_penalties.extend(kl_penalty.tolist())
        
        # Enhanced episode data for multi-turn logging
        episode_data = {
            'episode': episode,
            'prompts': prompts,
            'targets': targets,
            'thinking_contents': episode_thinking_contents,
            'contents': episode_contents,
            'rewards': rewards.tolist(),
            'base_rewards': base_rewards.tolist(),
            'penalties': penalties.tolist(),
            'thinking_penalties': thinking_penalties.tolist(),
            'loss': loss.item() * gradient_accumulation_steps,
            'kl_penalty_mean': kl_penalty.mean().item(),
            # Multi-turn specific data
            'turn_count': len(episode_commands),
            'episode_complete': episode_complete,
            'final_reward': final_reward,
            'commands': episode_commands,
            'command_outputs': episode_outputs,
            'terminal_context': episode_state.get('terminal_context', ''),
            'episode_rewards': episode_rewards,
            # Add full conversation context for debugging
            'all_model_outputs': all_model_outputs,
            'all_terminal_contexts': all_terminal_contexts,
            'conversation_dialogue': conversation_dialogue,  # The complete human-assistant conversation
            'episode_state': {
                'terminal_context': episode_state.get('terminal_context', ''),
                'turn_count': episode_state.get('turn_count', 0)
            }
        }
        
        accumulated_episodes.append(episode_data)
        
        # Perform optimizer step and logging at the end of accumulation
        if (episode + 1) % gradient_accumulation_steps == 0 or episode == config.num_episodes - 1:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Zero out special-token gradients
            zero_special_token_grads(model, tokenizer)
            optimizer.step()
            
            # Log accumulated metrics
            avg_loss = accumulated_loss / len(accumulated_episodes) if accumulated_episodes else 0.0
            avg_reward = sum(accumulated_rewards) / len(accumulated_rewards) if accumulated_rewards else 0.0
            avg_base_reward = sum(accumulated_base_rewards) / len(accumulated_base_rewards) if accumulated_base_rewards else 0.0
            avg_penalty = sum(accumulated_penalties) / len(accumulated_penalties) if accumulated_penalties else 0.0
            avg_thinking_penalty = sum(accumulated_thinking_penalties) / len(accumulated_thinking_penalties) if accumulated_thinking_penalties else 0.0
            avg_advantage = sum(accumulated_advantages) / len(accumulated_advantages) if accumulated_advantages else 0.0
            avg_kl_penalty = sum(accumulated_kl_penalties) / len(accumulated_kl_penalties) if accumulated_kl_penalties else 0.0
            
            # Calculate multi-turn specific averages
            avg_turn_count = sum(ep['turn_count'] for ep in accumulated_episodes) / len(accumulated_episodes) if accumulated_episodes else 0.0
            completion_rate = sum(1 for ep in accumulated_episodes if ep['episode_complete']) / len(accumulated_episodes) if accumulated_episodes else 0.0
            
            wandb_logger.log(
                {
                    "loss": avg_loss,
                    "reward_mean": avg_base_reward,
                    "penalized_reward": avg_reward,
                    "penalty": avg_penalty,
                    "thinking_penalty_mean": avg_thinking_penalty,
                    "advantage_mean": avg_advantage,
                    "kl_penalty_mean": avg_kl_penalty,
                    "kl_penalty_scaled": (config.kl_coefficient * avg_kl_penalty),
                    "gradient_accumulation_step": episode // gradient_accumulation_steps,
                    # Multi-turn specific metrics
                    "avg_turn_count": avg_turn_count,
                    "completion_rate": completion_rate,
                    "max_turns": max_turns,
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
                    thinking_penalties=episode_data['thinking_penalties'],
                    output_word_penalties=[{}],  # Not used in multi-turn
                    thinking_word_penalties=[{}],  # Not used in multi-turn
                    output_word_counts=[{}],  # Not used in multi-turn
                    thinking_word_counts=[{}],  # Not used in multi-turn
                    kl_penalty_mean=episode_data['kl_penalty_mean'],
                    # Multi-turn specific data
                    turn_count=episode_data['turn_count'],
                    episode_complete=episode_data['episode_complete'],
                    final_reward=episode_data['final_reward'],
                    commands=episode_data['commands'],
                    command_outputs=episode_data['command_outputs'],
                    terminal_context=episode_data['terminal_context'],
                    episode_rewards=episode_data['episode_rewards'],
                    conversation_dialogue=episode_data['conversation_dialogue'],
                )
            
            # Reset accumulation variables
            accumulated_loss = 0.0
            accumulated_rewards = []
            accumulated_base_rewards = []
            accumulated_penalties = []
            accumulated_thinking_penalties = []
            accumulated_advantages = []
            accumulated_kl_penalties = []
            accumulated_episodes = []

def train(config_path: str = "config.yaml") -> None:
    with open(config_path, "r") as f:
        config = Config(yaml.safe_load(f))

    # Check if multi-turn mode is enabled
    if getattr(config, 'enable_multi_turn', False):
        print("Multi-turn mode enabled, using multi-turn training function")
        return train_multi_turn(config_path)

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
    task = load_task(config.task_name, config.custom_verifier_path, config)
    wandb_logger   = WandbLogger(config)
    rollout_logger = RolloutLogger(config)

    # Load custom prompt at the start --------------------------------------
    custom_prompt = None
    if hasattr(config, 'custom_prompt_path') and config.custom_prompt_path:
        if config.custom_prompt_path == "registry":
            # Use the registry system
            try:
                from prompts.registry import registry
                # Get terminal configuration from config
                use_terminal = getattr(config, 'use_terminal', False)
                enable_multi_turn = getattr(config, 'enable_multi_turn', False)
                
                custom_prompt = registry.get_prompt(config.task_name, use_terminal=use_terminal, enable_multi_turn=enable_multi_turn)
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
    accumulated_base_rewards = []
    accumulated_penalties = []
    accumulated_thinking_penalties = []
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
            # Get terminal configuration for metadata
            use_terminal = getattr(config, 'use_terminal', False)
            enable_multi_turn = getattr(config, 'enable_multi_turn', False)
            
            prompts = [
                custom_prompt(b["question"], examples=None, metadata={
                    "task_name": config.task_name,
                    "use_terminal": use_terminal,
                    "enable_multi_turn": enable_multi_turn
                })
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
        
        # Prepend terminal instructions if needed
        if getattr(config, 'use_terminal', False):
            prompts = [prepend_terminal_instructions(p) for p in prompts]
            
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

        # Calculate rewards and penalties separately
        base_rewards = []
        penalties = []
        thinking_penalties = []
        penalized_rewards = []
        # For logging individual word penalties
        output_word_penalties = []
        thinking_word_penalties = []
        # For logging raw word counts
        output_word_counts = []
        thinking_word_counts = []
        
        for b, c, thinking_c in zip(batch, contents, thinking_contents):
            # Handle terminal tasks differently
            if hasattr(task, 'process_model_output'):
                # This is a terminal task
                result = task.process_model_output(c, b['answer'])
                base_reward = result['reward']
                penalty = 0.0
                thinking_penalty_for_logging = 0.0
                output_penalty_for_logging = 0.0
                output_word_penalties_dict = {}
                thinking_word_penalties_dict = {}
                output_word_counts_dict = {}
                thinking_word_counts_dict = {}
                penalized_reward = base_reward
                
                # Store terminal-specific metrics for later addition to episode data
                terminal_metrics = {}
                if 'terminal_context' in result:
                    terminal_metrics['terminal_context'] = result['terminal_context']
                if 'command_output' in result:
                    terminal_metrics['command_output'] = result['command_output']
                if 'verifier_used' in result:
                    terminal_metrics['verifier_used'] = result['verifier_used']
                if 'commands_executed' in result:
                    terminal_metrics['commands_executed'] = result['commands_executed']
            else:
                # Get the verifier to calculate base reward and penalty separately
                if hasattr(task, 'score_answer') and hasattr(task.score_answer, 'calculate_word_penalty'):
                    verifier = task.score_answer
                    base_reward = verifier.verify(c, b)
                    penalty = verifier.calculate_word_penalty(c)
                    
                    # Calculate penalties for logging (even when disabled)
                    output_penalty_for_logging = 0.0
                    thinking_penalty_for_logging = 0.0
                    output_word_penalties_dict = {}
                    thinking_word_penalties_dict = {}
                    output_word_counts_dict = {}
                    thinking_word_counts_dict = {}
                    
                    if hasattr(verifier, 'calculate_word_penalty_for_logging'):
                        output_penalty_for_logging = verifier.calculate_word_penalty_for_logging(c)
                    if hasattr(verifier, 'calculate_thinking_penalty_for_logging'):
                        thinking_penalty_for_logging = verifier.calculate_thinking_penalty_for_logging(thinking_c)
                    if hasattr(verifier, 'calculate_individual_word_penalties'):
                        output_word_penalties_dict = verifier.calculate_individual_word_penalties(c)
                    if hasattr(verifier, 'calculate_individual_thinking_word_penalties'):
                        thinking_word_penalties_dict = verifier.calculate_individual_thinking_word_penalties(thinking_c)
                    if hasattr(verifier, 'calculate_raw_word_counts'):
                        output_word_counts_dict = verifier.calculate_raw_word_counts(c)
                    if hasattr(verifier, 'calculate_raw_thinking_word_counts'):
                        thinking_word_counts_dict = verifier.calculate_raw_thinking_word_counts(thinking_c)
                    
                    penalized_reward = base_reward - penalty
                else:
                    base_reward = task.score_answer(c, b)
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
        
        # Convert to tensors
        base_rewards = torch.tensor(base_rewards, dtype=torch.float32, device=device)
        penalties = torch.tensor(penalties, dtype=torch.float32, device=device)
        thinking_penalties = torch.tensor(thinking_penalties, dtype=torch.float32, device=device)
        rewards = torch.tensor(penalized_rewards, dtype=torch.float32, device=device)  # Use penalized rewards for training

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
        accumulated_base_rewards.extend(base_rewards.tolist())
        accumulated_penalties.extend(penalties.tolist())
        accumulated_thinking_penalties.extend(thinking_penalties.tolist())
        accumulated_advantages.extend(advantage.tolist())
        accumulated_kl_penalties.extend(kl_penalty.tolist())
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
            'output_word_penalties': output_word_penalties,
            'thinking_word_penalties': thinking_word_penalties,
            'output_word_counts': output_word_counts,
            'thinking_word_counts': thinking_word_counts,
            'loss': loss.item() * gradient_accumulation_steps,  # Scale back for logging
            'kl_penalty_mean': kl_penalty.mean().item(),
        }
        
        # Add terminal-specific metrics if they exist
        if 'terminal_metrics' in locals():
            episode_data.update(terminal_metrics)
        
        accumulated_episodes.append(episode_data)

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
            avg_base_reward = sum(accumulated_base_rewards) / len(accumulated_base_rewards) if accumulated_base_rewards else 0.0
            avg_penalty = sum(accumulated_penalties) / len(accumulated_penalties) if accumulated_penalties else 0.0
            avg_thinking_penalty = sum(accumulated_thinking_penalties) / len(accumulated_thinking_penalties) if accumulated_thinking_penalties else 0.0
            avg_advantage = sum(accumulated_advantages) / len(accumulated_advantages) if accumulated_advantages else 0.0
            avg_kl_penalty = sum(accumulated_kl_penalties) / len(accumulated_kl_penalties) if accumulated_kl_penalties else 0.0
            avg_gen_tokens = float(mask.sum().item() / config.batch_size)
            
            # Calculate individual word penalty averages for logging
            all_output_word_penalties = {}
            all_thinking_word_penalties = {}
            all_output_word_counts = {}
            all_thinking_word_counts = {}
            for ep in accumulated_episodes:
                for word_penalties in ep['output_word_penalties']:
                    for word, penalty in word_penalties.items():
                        if word not in all_output_word_penalties:
                            all_output_word_penalties[word] = []
                        all_output_word_penalties[word].append(penalty)
                for word_penalties in ep['thinking_word_penalties']:
                    for word, penalty in word_penalties.items():
                        if word not in all_thinking_word_penalties:
                            all_thinking_word_penalties[word] = []
                        all_thinking_word_penalties[word].append(penalty)
                for word_counts in ep['output_word_counts']:
                    for word, count in word_counts.items():
                        if word not in all_output_word_counts:
                            all_output_word_counts[word] = []
                        all_output_word_counts[word].append(count)
                for word_counts in ep['thinking_word_counts']:
                    for word, count in word_counts.items():
                        if word not in all_thinking_word_counts:
                            all_thinking_word_counts[word] = []
                        all_thinking_word_counts[word].append(count)
            
            # Calculate averages for each word
            avg_output_word_penalties = {}
            avg_thinking_word_penalties = {}
            avg_output_word_counts = {}
            avg_thinking_word_counts = {}
            for word, penalties in all_output_word_penalties.items():
                avg_output_word_penalties[f"output_penalty_{word}"] = sum(penalties) / len(penalties)
            for word, penalties in all_thinking_word_penalties.items():
                avg_thinking_word_penalties[f"thinking_penalty_{word}"] = sum(penalties) / len(penalties)
            for word, counts in all_output_word_counts.items():
                avg_output_word_counts[f"output_count_{word}"] = sum(counts) / len(counts)
            for word, counts in all_thinking_word_counts.items():
                avg_thinking_word_counts[f"thinking_count_{word}"] = sum(counts) / len(counts)
            
            # Calculate total counts across all words
            total_output_words = sum(sum(counts) for counts in all_output_word_counts.values())
            total_thinking_words = sum(sum(counts) for counts in all_thinking_word_counts.values())
            
            wandb_logger.log(
                {
                    "loss": avg_loss,
                    "reward_mean": avg_base_reward,  # Base task reward (before penalty)
                    "penalized_reward": avg_reward,  # Final reward (after penalty)
                    "penalty": avg_penalty,  # Just the penalty amount
                    "thinking_penalty_mean": avg_thinking_penalty,
                    "advantage_mean": avg_advantage,
                    "kl_penalty_mean": avg_kl_penalty,
                    "kl_penalty_scaled": (config.kl_coefficient * avg_kl_penalty),
                    "avg_gen_tokens": avg_gen_tokens,
                    "gradient_accumulation_step": episode // gradient_accumulation_steps,
                    "total_output_penalized_words": total_output_words,
                    "total_thinking_penalized_words": total_thinking_words,
                    **avg_output_word_penalties,
                    **avg_thinking_word_penalties,
                    **avg_output_word_counts,
                    **avg_thinking_word_counts,
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
                    thinking_penalties=episode_data['thinking_penalties'],
                    output_word_penalties=episode_data['output_word_penalties'],
                    thinking_word_penalties=episode_data['thinking_word_penalties'],
                    output_word_counts=episode_data['output_word_counts'],
                    thinking_word_counts=episode_data['thinking_word_counts'],
                    kl_penalty_mean=episode_data['kl_penalty_mean'],
                )

            # Reset accumulation variables
            accumulated_loss = 0.0
            accumulated_rewards = []
            accumulated_base_rewards = []
            accumulated_penalties = []
            accumulated_thinking_penalties = []
            accumulated_advantages = []
            accumulated_kl_penalties = []
            accumulated_episodes = []

    wandb_logger.finish()

    # ----------------------------------------------------------------------
    # Optionally push the trained model (or LoRA adapters) to HuggingFace Hub
    # ----------------------------------------------------------------------
    repo_out = getattr(config, 'hf_repo_out', None)
    if repo_out:
        try:
            print(f"Pushing model to HuggingFace Hub repo '{repo_out}' ...")
            # Move model to CPU to free GPU memory before pushing
            model.cpu()
            # If the model is a PEFT model (e.g. LoRA), this will upload only the adapters.
            # Otherwise, it will upload the full model weights.
            model.push_to_hub(repo_out)
            # Upload the tokenizer to the same repository for completeness
            try:
                tokenizer.push_to_hub(repo_out)
            except Exception as tok_err:
                print(f"Tokenizer push failed: {tok_err}")
            print("Successfully pushed to HuggingFace Hub.")
        except Exception as push_err:
            print(f"Failed to push model to HuggingFace Hub: {push_err}")

def prepend_terminal_instructions(prompt: str) -> str:
    return (
        'You are in a restricted terminal environment. You have access to one directory containing a single file: "verifier.py". Your task is to pass the verifier.\n\n'
        'Terminal commands must be enclosed in <command></command> tags. For example:\n'
        '- <command>echo "Hello world"</command>\n'
        '- <command>python verifier.py "answer"</command>\n\n'
        'Anything outside of <command></command> tags will not be recognized as a command.\n\n'
        'Generate one command at a time. You will see the output of each command before generating the next one.\n\n'
        f'This is the problem the verifier checks solutions for: {prompt}'
    )