import torch
from shoggoth_face import ShoggothFace

def _extract_generations(output, tokenizer):
    """Reconstruct CoT vs. answer text from masks to demonstrate correctness."""
    ids = output["sequences"][0]  # (T,)
    prompt_len = output["prompt_lens"][0].item()
    pad_len = output["pad_lens"][0]
    effective_prompt_len = prompt_len + pad_len

    gen_tokens = ids[effective_prompt_len:]
    think_slice = output["think_mask"][0, effective_prompt_len - 1 : effective_prompt_len - 1 + gen_tokens.size(0)]
    face_slice  = output["face_mask"][0, effective_prompt_len - 1 : effective_prompt_len - 1 + gen_tokens.size(0)]

    think_tokens = gen_tokens[think_slice]
    face_tokens  = gen_tokens[face_slice]

    think_text = tokenizer.decode(think_tokens, skip_special_tokens=True).strip()
    face_text  = tokenizer.decode(face_tokens, skip_special_tokens=True).strip()
    return think_text, face_text


def test_shoggoth_face_multi_turn():
    # Instantiate ShoggothFace with the same model for shoggoth and face
    model_name = "Qwen/Qwen3-4B"
    sf = ShoggothFace(
        shoggoth_model_name=model_name,
        face_model_name=model_name,
        max_thinking_tokens=64,
        min_thinking_tokens=0,
        batch_size=1,
    )

    # Turn 1 ---------------------------------------------------------------
    user_msg_1 = "What is the capital of France?"

    # Use tokenizer.apply_chat_template to build the chat prompt
    messages_turn1 = [
        {"role": "user", "content": user_msg_1}
    ]
    prompt_1 = sf.tokenizer.apply_chat_template(
        messages_turn1,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    out1 = sf.generate([prompt_1], max_thinking_tokens=64, max_new_tokens=128)
    think1, face1 = _extract_generations(out1, sf.tokenizer)

    # Turn 2 ---------------------------------------------------------------
    user_msg_2 = "Why is it considered an important city historically?"

    # Build messages for second turn: user -> assistant -> user
    messages_turn2 = [
        {"role": "user", "content": user_msg_1},
        {"role": "assistant", "content": face1},
        {"role": "user", "content": user_msg_2},
    ]
    prompt_2 = sf.tokenizer.apply_chat_template(
        messages_turn2,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    out2 = sf.generate([prompt_2], max_thinking_tokens=64, max_new_tokens=128)
    think2, face2 = _extract_generations(out2, sf.tokenizer)

    # Print full exchange
    print("=== Conversation ===")
    print(f"User 1: {user_msg_1}")
    print(f"Assistant 1: {face1}")
    print(f"User 2: {user_msg_2}")
    print(f"Assistant 2: {face2}")

    # Print shoggoth / face parts separately per turn
    print("\n--- Shoggoth (CoT) vs Face outputs ---")
    print("Turn 1 – Shoggoth thinking:")
    print(think1 or "<empty>")
    print("Turn 1 – Face answer:")
    print(face1)

    print("\nTurn 2 – Shoggoth thinking:")
    print(think2 or "<empty>")
    print("Turn 2 – Face answer:")
    print(face2)


if __name__ == "__main__":
    test_shoggoth_face_multi_turn() 