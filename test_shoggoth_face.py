"""Test for the ShoggothFace generation wrapper.

This script loads both *shoggoth* and *face* models from the same
`Qwen/Qwen3-0.6B` repository (you may change the constants), runs them on
example prompts, and writes the decoded chain-of-thought (shoggoth) and
visible answer (face) to a JSONL file.

Usage
-----
    python test_shoggoth_face.py
The output file path is controlled by the OUTPUT_PATH constant below.
"""

import json
import os
import torch

from models.qwen3 import prepare_thinking_input
from shoggoth_face import ShoggothFace

# ---------------------------------------------------------------------
# Global parameters – tweak as needed
# ---------------------------------------------------------------------
SHOGGOTH_MODEL_NAME = "Qwen/Qwen3-0.6B"
FACE_MODEL_NAME     = "Qwen/Qwen3-0.6B"  # can differ from SHOGGOTH_MODEL_NAME
PROMPTS             = [
    "What is 2 + 2?",
    "Translate the English word 'Hello' to French.",
    "Name the capital city of Japan.",
]
MAX_THINKING_TOKENS = 64
MAX_NEW_TOKENS      = 128  # total (thinking + answer)
OUTPUT_PATH         = "shoggoth_face_test_output.jsonl"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE          = len(PROMPTS)


def main() -> None:
    # Build the generation engine (models load internally)
    gen_engine = ShoggothFace(
        shoggoth_model_name=SHOGGOTH_MODEL_NAME,
        face_model_name=FACE_MODEL_NAME,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        max_thinking_tokens=MAX_THINKING_TOKENS,
        min_thinking_tokens=0,
    )

    tokenizer = gen_engine.tokenizer

    # Prepare prompts with <think> tag placeholders
    prompt_inputs = [prepare_thinking_input(tokenizer, p, enable_thinking=True) for p in PROMPTS]

    # Generate
    with torch.no_grad():
        gen_dict = gen_engine.generate(
            prompt_inputs,
            max_thinking_tokens=MAX_THINKING_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    shoggoth_gens = gen_dict["shoggoth_generations"]
    face_gens     = gen_dict["face_generations"]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    # Write JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for prompt, shog, face in zip(PROMPTS, shoggoth_gens, face_gens):
            rec = {"prompt": prompt, "shoggoth": shog, "face": face}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(PROMPTS)} generations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main() 