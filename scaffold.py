"""
Tiny GPT From Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Tiny GPT from Scratch in NumPy — end-to-end demo scaffold."""

import numpy as np

from solution import (
    build_char_vocab,
    build_id_to_char,
    encode_text,
    decode_ids,
    make_batches,
    token_embedding_lookup,
    add_positional_embeddings,
    linear_projection,
    compute_attention_scores,
    scale_attention_scores,
    apply_causal_mask,
    softmax_attention_weights,
    attention_context,
    split_heads,
    merge_heads,
    project_qkv,
    multi_head_scaled_dot_product_attention,
    merge_and_output_project,
    masked_multi_head_self_attention,
    gelu_activation,
    ffn_first_layer,
    ffn_second_layer,
    position_wise_feed_forward,
    layernorm_stats,
    layer_norm,
    pre_norm_residual_sublayer,
    transformer_block,
    gpt_backbone,
    project_to_vocab_logits,
    gpt_forward,
    cross_entropy_language_modeling_loss,
    init_gpt_parameters,
    collect_parameters,
    training_step,
    apply_optimizer_update,
    run_training_loop,
    last_position_logits,
    scale_logits_by_temperature,
    top_k_filter_logits,
    sample_next_token,
    generate_text,
)


def main():
    np.random.seed(0)
    rng = np.random.default_rng(0)

    corpus = (
        "hello world. this is a tiny gpt demo built from numpy.\n"
        "the quick brown fox jumps over the lazy dog.\n"
        "transformers learn to predict the next character.\n"
    ) * 8

    char_to_id = build_char_vocab(corpus)
    id_to_char = build_id_to_char(char_to_id)
    vocab_size = len(char_to_id)
    data = encode_text(corpus, char_to_id)
    print(f"vocab_size={vocab_size}, corpus_tokens={len(data)}")
    print(f"sample decode: {decode_ids(data[:30], id_to_char)!r}")

    block_size = 16
    batch_size = 4
    d_model = 32
    num_heads = 4
    d_ff = 64
    num_layers = 2
    num_steps = 3
    learning_rate = 1e-2

    params = init_gpt_parameters(
        vocab_size=vocab_size,
        max_seq_len=block_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        seed=0,
    )
    # Stash num_heads so training_step / generate_text can find it.
    params['num_heads'] = num_heads

    flat_params = collect_parameters(params)
    print(f"num parameter arrays: {len(flat_params)}")

    inputs, targets = make_batches(data, batch_size, block_size, rng)
    print(f"batch shapes: inputs={inputs.shape}, targets={targets.shape}")
    logits = gpt_forward(inputs, params, num_heads)
    print(f"logits shape: {logits.shape}")
    init_loss = cross_entropy_language_modeling_loss(logits, targets)
    print(f"initial loss: {init_loss:.4f}")

    batches = [make_batches(data, batch_size, block_size, rng) for _ in range(num_steps)]
    losses = run_training_loop(params, batches, num_steps, learning_rate)
    print(f"loss history (first/last): {losses[0]:.4f} -> {losses[-1]:.4f}")

    prompt = "the "
    generated = generate_text(
        params=params,
        prompt=prompt,
        num_new_tokens=40,
        max_seq_len=block_size,
        vocab=char_to_id,
        id_to_char=id_to_char,
        temperature=1.0,
        top_k=5,
        seed=0,
    )
    print(f"prompt: {prompt!r}")
    print(f"generated: {generated!r}")


if __name__ == "__main__":
    main()
