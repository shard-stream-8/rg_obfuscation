import torch
from transformers import LogitsProcessor

class BatchThinkingTokenBudgetProcessor(LogitsProcessor):
    """Optimized thinking token processor that handles batched generation."""
    def __init__(self, tokenizer, max_thinking_tokens=None, batch_size=8, min_thinking_tokens=0):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.min_thinking_tokens = min_thinking_tokens
        self.think_end_tokens = self.tokenizer.encode("</think>", add_special_tokens=False)
        self.nl_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
        self.batch_size = batch_size
        self.tokens_generated = [0] * batch_size
        self.stopped_thinking = [False] * batch_size
        self.neg_inf = -1e10

    def reset(self):
        """Reset the processor state for a new episode."""
        self.tokens_generated = [0] * self.batch_size
        self.stopped_thinking = [False] * self.batch_size

    def _set_token_score(self, scores, token_ids, value, batch_idx):
        for tid in token_ids:
            if tid < scores.shape[1]:
                scores[batch_idx][tid] = value
                if value == 0.0:
                    scores[batch_idx][tid] = 1.0

    def _set_all_scores_to_neg_inf(self, scores, batch_idx):
        scores[batch_idx][:] = self.neg_inf

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0]
        for batch_idx in range(batch_size):
            if batch_idx >= len(self.tokens_generated):
                self.tokens_generated.extend([0] * (batch_size - len(self.tokens_generated)))
                self.stopped_thinking.extend([False] * (batch_size - len(self.stopped_thinking)))
            self.tokens_generated[batch_idx] += 1
            if self.max_thinking_tokens == 0 and not self.stopped_thinking[batch_idx] and self.tokens_generated[batch_idx] > 0:
                self._set_all_scores_to_neg_inf(scores, batch_idx)
                self._set_token_score(scores, self.nl_tokens, 0.0, batch_idx)
                self._set_token_score(scores, self.think_end_tokens, 0.0, batch_idx)
                self.stopped_thinking[batch_idx] = True
            elif self.max_thinking_tokens is not None and not self.stopped_thinking[batch_idx]:
                if (self.max_thinking_tokens > 0 and self.tokens_generated[batch_idx] / self.max_thinking_tokens) > 0.8:
                    boost_factor = 1.0 + (self.tokens_generated[batch_idx] / self.max_thinking_tokens)
                    for tid in self.nl_tokens:
                        if tid < scores.shape[1]:
                            scores[batch_idx][tid] *= boost_factor
                    for tid in self.think_end_tokens:
                        if tid < scores.shape[1]:
                            scores[batch_idx][tid] *= boost_factor
                if self.max_thinking_tokens > 0 and self.tokens_generated[batch_idx] == self.max_thinking_tokens - 2:
                    self._set_all_scores_to_neg_inf(scores, batch_idx)
                    self._set_token_score(scores, self.nl_tokens, 0.0, batch_idx)
                elif self.max_thinking_tokens > 0 and self.tokens_generated[batch_idx] >= self.max_thinking_tokens - 1:
                    self._set_all_scores_to_neg_inf(scores, batch_idx)
                    self._set_token_score(scores, self.think_end_tokens, 0.0, batch_idx)
                    self.stopped_thinking[batch_idx] = True
            if not self.stopped_thinking[batch_idx] and self.tokens_generated[batch_idx] < self.min_thinking_tokens:
                for tid in self.think_end_tokens:
                    scores[batch_idx][tid] = self.neg_inf
        return scores 

# -----------------------------------------------------------------------------
# NEW: Processor to enforce a minimum number of answer tokens after </think>
# -----------------------------------------------------------------------------
class MinOutputTokensProcessor(LogitsProcessor):
    """Disallow EOS until *min_output_tokens* tokens have been generated *after* the
    first </think> token. This prevents the model from immediately terminating
    its visible answer once it has finished thinking.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer whose special tokens are used to identify </think> and EOS.
    min_output_tokens : int | None
        Required number of tokens after </think>. If ``None`` or ``0`` the
        processor becomes a no-op.
    batch_size : int, default 8
        Used to size internal bookkeeping arrays.
    """

    def __init__(self, tokenizer, min_output_tokens=None, batch_size: int = 8):
        self.tokenizer = tokenizer
        self.min_output_tokens = (
            None if min_output_tokens in (None, "null") else int(min_output_tokens)
        )
        self.batch_size = batch_size
        self.end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.eos_token_id = tokenizer.eos_token_id
        # Fail fast if EOS token is unavailable – the processor cannot operate without it
        if self.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id; MinOutputTokensProcessor requires a defined EOS token.")
        self.neg_inf = -1e10

    def reset(self):
        """No persistent state is required but keep for API symmetry."""
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Fast-exit when disabled
        if not self.min_output_tokens or self.min_output_tokens <= 0:
            return scores

        batch_size = input_ids.size(0)
        vocab_size = scores.size(1)
        # Ensure eos_token_id is within the model's vocabulary
        if self.eos_token_id >= vocab_size:
            raise ValueError(
                f"eos_token_id ({self.eos_token_id}) is outside the model's vocabulary (size {vocab_size})."
            )

        for batch_idx in range(batch_size):
            seq = input_ids[batch_idx]
            # Locate last </think> token in the sequence so far
            try:
                end_pos = (seq == self.end_think_token_id).nonzero(as_tuple=False)[-1].item()
            except IndexError:
                # </think> not yet emitted – nothing to enforce
                continue

            tokens_after = seq.size(0) - end_pos - 1  # how many tokens after </think>
            if tokens_after < self.min_output_tokens:
                scores[batch_idx][self.eos_token_id] = self.neg_inf
        return scores 