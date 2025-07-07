import torch

def zero_special_token_grads(model, tokenizer):
    special_tokens = ["<think>", "</think>", "\n"]
    token_ids = []
    for tok in special_tokens:
        token_ids.extend(tokenizer.encode(tok, add_special_tokens=False))
    if hasattr(model, 'get_output_embeddings'):
        emb = model.get_output_embeddings()
        if hasattr(emb, 'weight') and emb.weight.grad is not None:
            for tid in token_ids:
                if tid < emb.weight.grad.shape[0]:
                    emb.weight.grad[tid].zero_() 