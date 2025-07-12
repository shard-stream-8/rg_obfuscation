import torch
from transformers import LogitsProcessorList


# Local imports
from models.qwen3 import load_qwen3_model
from reinforce.logit_processor import BatchThinkingTokenBudgetProcessor
from reinforce.logit_processor import MinOutputTokensProcessor


class ShoggothFace:
    """Wrapper that lets a *shoggoth* model generate the chain-of-thought
    (between <think> and </think>) while a *face* model generates the final
    visible answer.

    Parameters
    ----------
    shoggoth_model_name : str
        HF repo path for the model that will produce the chain-of-thought.
    face_model_name : str
        HF repo path for the model that will produce the final answer.
    device : str, optional
        Torch device spec.  Defaults to "cuda" if available otherwise "cpu".
    batch_size : int
        The batch size to expect when generating. Used to size internal
        `BatchThinkingTokenBudgetProcessor`.
    max_thinking_tokens : int, optional
        Maximum number of <think> tokens allowed. Required if
        `logit_processor` is not supplied.
    min_thinking_tokens : int, optional
        Minimum number of <think> tokens.  Defaults to 0.
    logit_processor : transformers.LogitsProcessor or list, optional
        Custom logits processor for the thinking phase. If omitted, an
        instance of `BatchThinkingTokenBudgetProcessor` is created using the
        provided token budgets.
    """

    def __init__(
        self,
        shoggoth_model_name: str,
        face_model_name: str,
        *,
        device: str = None,
        batch_size: int = 8,
        max_thinking_tokens: int | None = None,
        min_thinking_tokens: int = 0,
        min_output_tokens: int | None = None,
        logit_processor=None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Informative prints before model loading
        print(f"[INFO] Loading shoggoth model: {shoggoth_model_name}")
        print(f"[INFO] Loading face model: {face_model_name}")
        # Load models & tokeniser --------------------------------------------------
        self.face_model, face_tokenizer, _ = load_qwen3_model(face_model_name, device)
        self.shoggoth_model, shoggoth_tokenizer, _ = load_qwen3_model(shoggoth_model_name, device)

        # Assert tokeniser identity ------------------------------------------------
        if face_tokenizer.get_vocab() != shoggoth_tokenizer.get_vocab():
            raise ValueError("Tokenizers of face and shoggoth models differ – they must share the same vocabulary")

        self.tokenizer = face_tokenizer
        self.device = device

        # Build / wrap logits processor -------------------------------------------
        if logit_processor is None:
            # Need max_thinking_tokens and batch_size
            if max_thinking_tokens is None:
                raise ValueError("max_thinking_tokens must be provided when logit_processor is None")
            self.logit_processor = LogitsProcessorList([
                BatchThinkingTokenBudgetProcessor(
                    self.tokenizer,
                    max_thinking_tokens=max_thinking_tokens,
                    batch_size=batch_size,
                    min_thinking_tokens=min_thinking_tokens,
                )
            ])
        elif isinstance(logit_processor, list):
            self.logit_processor = LogitsProcessorList(logit_processor)
        else:
            self.logit_processor = LogitsProcessorList([logit_processor])

        # Store answer-length constraint
        self.min_output_tokens = min_output_tokens

    @torch.no_grad()
    def generate(self, prompt_inputs, max_thinking_tokens, max_new_tokens, **gen_kwargs):
        """Generate full responses for *prompt_inputs*.

        Args:
            prompt_inputs (list[str]): list of already-templated prompts that
                include the generation placeholder for chat models.
            max_thinking_tokens (int): budget for CoT tokens (handled by the
                supplied logits processor).
            max_new_tokens (int): total budget (thinking + answer).
            **gen_kwargs: forwarded to both model.generate() calls.

        Returns:
            dict with keys:
                sequences: (B, T) tensor with prompt+CoT+answer ids
                think_mask: boolean mask same size as sequences identifying
                    positions whose *next* token was produced by the shoggoth
                face_mask: boolean mask identifying positions produced by the
                    face model (answer)
                prompt_lens: list[int] prompt lengths
                think_lens:  list[int] lengths of CoT (incl. </think>)
        """
        # ------------------------------------------------------------------
        # Stage 1: THINKING (shoggoth)
        # ------------------------------------------------------------------
        # Tokenise prompts once so the same padding is used for both models
        model_inputs = self.tokenizer(prompt_inputs, return_tensors="pt", padding=True)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        prompt_lens = (model_inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)

        # Generate up to max_thinking_tokens using the logits processor that
        # will force a closing </think> token when appropriate.
        shog_outputs = self.shoggoth_model.generate(
            **model_inputs,
            max_new_tokens=max_thinking_tokens,
            logits_processor=self.logit_processor,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        )
        shog_sequences = shog_outputs.sequences  # (B, prompt_len + think_len)

        # ------------------------------------------------------------------
        # Trim everything AFTER the first </think> token --------------------
        # ------------------------------------------------------------------
        end_think_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        trimmed_sequences = []
        think_lens = []
        for seq, p_len in zip(shog_sequences, prompt_lens):
            # work on python int list for convenience
            seq_list = seq.tolist()
            # search only in generated part to speedup slightly
            try:
                # position (absolute) of first end_think token AFTER prompt
                rel_idx = seq_list[p_len:].index(end_think_id)
                end_idx = p_len + rel_idx  # inclusive position of </think>
            except ValueError:
                # Should not happen because logits_processor forces it, but be
                # defensive: treat the entire generated segment as thinking.
                end_idx = len(seq_list) - 1
            kept = seq_list[: end_idx + 1]  # keep </think>
            trimmed_sequences.append(torch.tensor(kept, dtype=torch.long))
            think_lens.append(len(kept) - p_len)

        # Left-pad so that the </think> tokens align across the batch --------
        max_len = max(len(seq) for seq in trimmed_sequences)
        pad_id = self.tokenizer.pad_token_id
        padded_inputs = []
        attention_masks = []
        pad_lens = []  # store how much left-padding was added for each sample
        for seq in trimmed_sequences:
            pad_len = max_len - len(seq)
            pad_lens.append(pad_len)
            padded_seq = torch.cat([
                torch.full((pad_len,), pad_id, dtype=torch.long),
                seq
            ])
            padded_inputs.append(padded_seq)
            attention_masks.append((padded_seq != pad_id).long())
        input_ids_face = torch.stack(padded_inputs).to(self.device)
        attention_mask_face = torch.stack(attention_masks).to(self.device)

        # ------------------------------------------------------------------
        # Stage 2: ANSWER (face)
        # ------------------------------------------------------------------
        min_think_len = min(think_lens)
        face_max_new_tokens = max_new_tokens - min_think_len
        if face_max_new_tokens < 0:
            face_max_new_tokens = 0

        # Build logits processor for answer phase if minimum output tokens requested
        answer_processors = None
        if self.min_output_tokens is not None:
            answer_processors = LogitsProcessorList([
                MinOutputTokensProcessor(
                    self.tokenizer,
                    min_output_tokens=self.min_output_tokens,
                    batch_size=input_ids_face.size(0),
                )
            ])

        face_outputs = self.face_model.generate(
            input_ids=input_ids_face,
            attention_mask=attention_mask_face,
            max_new_tokens=face_max_new_tokens,
            logits_processor=answer_processors,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        )
        full_sequences = face_outputs.sequences  # (B, prompt+think+answer)

        # ------------------------------------------------------------------
        # Build masks -------------------------------------------------------
        # ------------------------------------------------------------------
        B, T = full_sequences.shape
        think_mask = torch.zeros((B, T - 1), dtype=torch.bool, device=self.device)
        face_mask = torch.zeros((B, T - 1), dtype=torch.bool, device=self.device)
        for b in range(B):
            p_len = prompt_lens[b].item()
            pad_len = pad_lens[b]
            t_len = think_lens[b]
            effective_prompt_len = p_len + pad_len  # prompt index after padding

            # token positions whose NEXT token belong to CoT (between prompt and answer)
            start = effective_prompt_len - 1  # index of token *before* first CoT token
            end = effective_prompt_len + t_len - 1  # last index before answer begins

            think_mask[b, start:end] = True
            face_mask[b, end:(T - 1)] = True

        # ------------------------------------------------------------------
        # Decode shoggoth & face generations using masks -------------------
        # ------------------------------------------------------------------
        shoggoth_generations = []
        face_generations = []

        for b in range(B):
            p_len = prompt_lens[b].item()
            pad_len = pad_lens[b]
            effective_prompt_len = p_len + pad_len
            gen_tokens = full_sequences[b, effective_prompt_len:]
            # slice relevant mask parts (length L)
            start_mask_idx = effective_prompt_len - 1
            think_slice = think_mask[b, start_mask_idx: start_mask_idx + gen_tokens.size(0)]
            face_slice = face_mask[b, start_mask_idx: start_mask_idx + gen_tokens.size(0)]

            shog_ids = gen_tokens[think_slice].tolist()
            face_ids = gen_tokens[face_slice].tolist()

            shoggoth_generations.append(
                self.tokenizer.decode(shog_ids, skip_special_tokens=True).strip()
            )
            face_generations.append(
                self.tokenizer.decode(face_ids, skip_special_tokens=True).strip()
            )

        return {
            "sequences": full_sequences,
            "prompt_lens": prompt_lens,
            "think_lens": think_lens,
            "think_mask": think_mask,
            "face_mask": face_mask,
            "pad_lens": pad_lens,
            "shoggoth_generations": shoggoth_generations,
            "face_generations": face_generations,
        } 