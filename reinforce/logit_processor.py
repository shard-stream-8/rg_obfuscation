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