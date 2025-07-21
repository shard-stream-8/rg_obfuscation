import torch

def zero_special_token_grads(model, tokenizer):
    special_tokens = ["<think>", "</think>", "\n"]
    token_ids = []
    for tok in special_tokens:
        token_ids.extend(tokenizer.encode(tok, add_special_tokens=False))

    # Zero output embedding gradients (as before)
    if hasattr(model, 'get_output_embeddings'):
        emb = model.get_output_embeddings()
        if hasattr(emb, 'weight') and emb.weight.grad is not None:
            for tid in token_ids:
                if tid < emb.weight.grad.shape[0]:
                    emb.weight.grad[tid].zero_()

    # Zero input embedding gradients (if present)
    if hasattr(model, 'get_input_embeddings'):
        emb_in = model.get_input_embeddings()
        if hasattr(emb_in, 'weight') and emb_in.weight.grad is not None:
            for tid in token_ids:
                if tid < emb_in.weight.grad.shape[0]:
                    emb_in.weight.grad[tid].zero_()

    # Zero any other parameter gradients that are indexed by vocab (rare, but for completeness)
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.shape[0] >= max(token_ids) + 1:
            # Only zero if the first dimension matches vocab size
            for tid in token_ids:
                if tid < param.grad.shape[0]:
                    param.grad[tid].zero_() 