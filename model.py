"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_char_vocab
def build_char_vocab(text):
    # Get unique characters, sort them, and assign IDs starting from 0
    return {char: idx for idx, char in enumerate(sorted(set(text)))}

# Step 2 - build_id_to_char
def build_id_to_char(char_to_id):
    return {idx: char for char, idx in char_to_id.items()}

# Step 3 - encode_text
import numpy as np

def encode_text(text, char_to_id):
    return np.array([char_to_id[char] for char in text], dtype=np.int64)

# Step 4 - decode_ids
def decode_ids(ids, id_to_char):
    return ''.join(id_to_char[int(idx)] for idx in ids)

# Step 5 - make_batches
def make_batches(data, batch_size, block_size, rng):
    starts = rng.integers(0, len(data) - block_size, size=batch_size)

    x = np.array(
        [data[start:start + block_size] for start in starts],
        dtype=np.int64
    )

    y = np.array(
        [data[start + 1:start + block_size + 1] for start in starts],
        dtype=np.int64
    )

    return x, y

# Step 6 - token_embedding_lookup
def token_embedding_lookup(token_ids, embedding_table):
    return embedding_table[token_ids]

# Step 7 - add_positional_embeddings
def add_positional_embeddings(token_embeds, pos_embedding_table):
    T = token_embeds.shape[1]
    return token_embeds + pos_embedding_table[:T]

# Step 8 - linear_projection
def linear_projection(x, weight, bias):
    output = x @ weight

    if bias is not None:
        output = output + bias

    return output

# Step 9 - compute_attention_scores
def compute_attention_scores(q, k):
    return q @ np.swapaxes(k, -1, -2)

# Step 10 - scale_attention_scores
def scale_attention_scores(scores, head_dim):
    return scores / np.sqrt(head_dim)

# Step 11 - apply_causal_mask (not yet solved)
# TODO: implement

# Step 12 - softmax_attention_weights (not yet solved)
# TODO: implement

# Step 13 - attention_context (not yet solved)
# TODO: implement

# Step 14 - split_heads (not yet solved)
# TODO: implement

# Step 15 - merge_heads (not yet solved)
# TODO: implement

# Step 16 - project_qkv (not yet solved)
# TODO: implement

# Step 17 - multi_head_scaled_dot_product_attention (not yet solved)
# TODO: implement

# Step 18 - merge_and_output_project (not yet solved)
# TODO: implement

# Step 19 - masked_multi_head_self_attention (not yet solved)
# TODO: implement

# Step 20 - gelu_activation (not yet solved)
# TODO: implement

# Step 21 - ffn_first_layer (not yet solved)
# TODO: implement

# Step 22 - ffn_second_layer (not yet solved)
# TODO: implement

# Step 23 - position_wise_feed_forward (not yet solved)
# TODO: implement

# Step 24 - layernorm_stats (not yet solved)
# TODO: implement

# Step 25 - layer_norm (not yet solved)
# TODO: implement

# Step 26 - pre_norm_residual_sublayer (not yet solved)
# TODO: implement

# Step 27 - transformer_block (not yet solved)
# TODO: implement

# Step 28 - gpt_backbone (not yet solved)
# TODO: implement

# Step 29 - project_to_vocab_logits (not yet solved)
# TODO: implement

# Step 30 - gpt_forward (not yet solved)
# TODO: implement

# Step 31 - cross_entropy_language_modeling_loss (not yet solved)
# TODO: implement

# Step 32 - init_gpt_parameters (not yet solved)
# TODO: implement

# Step 33 - collect_parameters (not yet solved)
# TODO: implement

# Step 34 - training_step (not yet solved)
# TODO: implement

# Step 35 - apply_optimizer_update (not yet solved)
# TODO: implement

# Step 36 - run_training_loop (not yet solved)
# TODO: implement

# Step 37 - last_position_logits (not yet solved)
# TODO: implement

# Step 38 - scale_logits_by_temperature (not yet solved)
# TODO: implement

# Step 39 - top_k_filter_logits (not yet solved)
# TODO: implement

# Step 40 - sample_next_token (not yet solved)
# TODO: implement

# Step 41 - generate_text (not yet solved)
# TODO: implement

