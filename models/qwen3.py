from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_qwen3_model(model_name, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer, device

def prepare_thinking_input(tokenizer, prompt, enable_thinking=True):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    return text 