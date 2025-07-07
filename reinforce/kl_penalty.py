import torch
import torch.nn.functional as F

def compute_kl_penalty(rewards_tensor, ref_model, config, full_ids, response_start_idx, response_logits, gradient_mask):
    kl_penalty = torch.zeros_like(rewards_tensor)
    if ref_model is not None and config and getattr(config, 'use_kl_penalty', False):
        with torch.no_grad():
            ref_logits = ref_model(full_ids, return_dict=True).logits[:, response_start_idx:-1]
            current_log_probs = F.log_softmax(response_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            current_probs = current_log_probs.exp()
            kl_div_per_token = F.kl_div(ref_log_probs, current_probs, reduction='none', log_target=True).sum(dim=-1)
            kl_div_per_token = kl_div_per_token * gradient_mask
            kl_div = kl_div_per_token.sum(dim=-1)
            sequence_lengths = gradient_mask.sum(dim=-1)
            kl_penalty = kl_div / (sequence_lengths + 1e-8)
    return kl_penalty 