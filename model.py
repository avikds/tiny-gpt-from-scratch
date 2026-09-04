"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    return sorted(set(text))

# Step 2 - build_stoi
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    return {ch: i for i, ch in enumerate(vocab)}

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    return {i: ch for i, ch in enumerate(vocab)}

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    return [encode_char(ch, stoi) for ch in text]

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    return ''.join(decode_int(token_id, itos) for token_id in ids)

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    return np.array(values)

# Step 9 - get_array_shape
def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    return arr.shape

# Step 10 - get_array_dtype
def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    return arr.dtype

# Step 11 - make_2d_zeros
def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    return np.zeros((rows, cols))

# Step 12 - make_2d_random
def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    return arr[i, j]

# Step 14 - slice_row
def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    return arr[i]

# Step 15 - slice_column
def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    return arr[:, j]

# Step 16 - slice_subblock
def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    return arr[r0:r1, c0:c1]

# Step 17 - elementwise_add
def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    return a + b

# Step 18 - elementwise_multiply
def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    return a * b

# Step 19 - scalar_broadcast_add
def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    return matrix + vector

# Step 21 - array_exp
def array_exp(arr):
    """Return the elementwise exponential of arr."""
    return np.exp(arr)

# Step 22 - array_log
def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    return np.log(arr)

# Step 23 - sum_all
def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    return arr.sum()

# Step 24 - sum_axis0
def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    return np.sum(arr, axis=0)

# Step 25 - sum_axis1
def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    return np.sum(arr, axis=1)

# Step 26 - max_along_axis
def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    return np.max(arr, axis=axis)

# Step 27 - matmul
def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    return a @ b

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    return arr.T

# Step 29 - sum_keepdims
def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    return np.sum(arr, axis=axis, keepdims=True)

# Step 30 - naive_softmax_1d
def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    exp_logits = array_exp(logits)
    total = sum_all(exp_logits)
    return exp_logits / total

# Step 31 - softmax_overflow_demo (not yet solved)
# TODO: implement

# Step 32 - stable_softmax_1d (not yet solved)
# TODO: implement

# Step 33 - stable_softmax_2d_rowwise (not yet solved)
# TODO: implement

# Step 34 - read_text_file (not yet solved)
# TODO: implement

# Step 35 - encode_corpus_to_int_array (not yet solved)
# TODO: implement

# Step 36 - pick_split_point (not yet solved)
# TODO: implement

# Step 37 - slice_train_and_val (not yet solved)
# TODO: implement

# Step 38 - pick_block_size (not yet solved)
# TODO: implement

# Step 39 - slice_x_at_offset (not yet solved)
# TODO: implement

# Step 40 - slice_y_at_offset (not yet solved)
# TODO: implement

# Step 41 - sample_random_batch_offsets (not yet solved)
# TODO: implement

# Step 42 - stack_x_batch (not yet solved)
# TODO: implement

# Step 43 - stack_y_batch (not yet solved)
# TODO: implement

# Step 44 - get_batch (not yet solved)
# TODO: implement

# Step 45 - allocate_count_matrix (not yet solved)
# TODO: implement

# Step 46 - loop_fill_counts (not yet solved)
# TODO: implement

# Step 47 - vectorize_counts_add_at (not yet solved)
# TODO: implement

# Step 48 - add_one_smoothing (not yet solved)
# TODO: implement

# Step 49 - row_sums_of_counts (not yet solved)
# TODO: implement

# Step 50 - normalize_counts_to_probs (not yet solved)
# TODO: implement

# Step 51 - sample_next_token (not yet solved)
# TODO: implement

# Step 52 - generate_sequence (not yet solved)
# TODO: implement

# Step 53 - decode_generated_sequence (not yet solved)
# TODO: implement

# Step 54 - log_prob_of_pair (not yet solved)
# TODO: implement

# Step 55 - sum_negative_log_probs (not yet solved)
# TODO: implement

# Step 56 - average_nll (not yet solved)
# TODO: implement

# Step 57 - initialize_w_random (not yet solved)
# TODO: implement

# Step 58 - scale_w_small (not yet solved)
# TODO: implement

# Step 59 - one_hot_encode_batch (not yet solved)
# TODO: implement

# Step 60 - forward_logits_onehot (not yet solved)
# TODO: implement

# Step 61 - observe_lookup_equivalence (not yet solved)
# TODO: implement

# Step 62 - forward_logits_lookup (not yet solved)
# TODO: implement

# Step 63 - logits_to_probs_rowwise (not yet solved)
# TODO: implement

# Step 64 - gather_correct_token_probs (not yet solved)
# TODO: implement

# Step 65 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 66 - derive_dlogits_on_paper (not yet solved)
# TODO: implement

# Step 67 - compute_dlogits (not yet solved)
# TODO: implement

# Step 68 - derive_dw_on_paper (not yet solved)
# TODO: implement

# Step 69 - compute_dw_scatter_add (not yet solved)
# TODO: implement

# Step 70 - sgd_update_w (not yet solved)
# TODO: implement

# Step 71 - run_one_training_step (not yet solved)
# TODO: implement

# Step 72 - train_neural_bigram_loop (not yet solved)
# TODO: implement

# Step 73 - sample_from_neural_bigram (not yet solved)
# TODO: implement

# Step 74 - linear_forward (not yet solved)
# TODO: implement

# Step 75 - derive_dx_on_paper (not yet solved)
# TODO: implement

# Step 76 - derive_linear_dw_on_paper (not yet solved)
# TODO: implement

# Step 77 - linear_backward_dx (not yet solved)
# TODO: implement

# Step 78 - linear_backward_dw (not yet solved)
# TODO: implement

# Step 79 - bias_add_forward (not yet solved)
# TODO: implement

# Step 80 - bias_add_backward_db (not yet solved)
# TODO: implement

# Step 81 - relu_forward (not yet solved)
# TODO: implement

# Step 82 - relu_backward (not yet solved)
# TODO: implement

# Step 83 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 84 - layernorm_forward_mean (not yet solved)
# TODO: implement

# Step 85 - layernorm_forward_variance (not yet solved)
# TODO: implement

# Step 86 - layernorm_forward_normalize (not yet solved)
# TODO: implement

# Step 87 - layernorm_forward_affine (not yet solved)
# TODO: implement

# Step 88 - layernorm_backward_subtract_mean (not yet solved)
# TODO: implement

# Step 89 - layernorm_backward_divide_std (not yet solved)
# TODO: implement

# Step 90 - layernorm_backward_full (not yet solved)
# TODO: implement

# Step 91 - layernorm_backward_implementation (not yet solved)
# TODO: implement

# Step 92 - create_token_embedding (not yet solved)
# TODO: implement

# Step 93 - token_embedding_forward (not yet solved)
# TODO: implement

# Step 94 - token_embedding_backward (not yet solved)
# TODO: implement

# Step 95 - create_positional_embedding (not yet solved)
# TODO: implement

# Step 96 - slice_positional_embedding (not yet solved)
# TODO: implement

# Step 97 - add_token_and_positional_embeddings (not yet solved)
# TODO: implement

# Step 98 - embedding_sum_backward (not yet solved)
# TODO: implement

# Step 99 - create_qkv_projections (not yet solved)
# TODO: implement

# Step 100 - compute_query (not yet solved)
# TODO: implement

# Step 101 - compute_key (not yet solved)
# TODO: implement

# Step 102 - compute_value (not yet solved)
# TODO: implement

# Step 103 - compute_attention_scores (not yet solved)
# TODO: implement

# Step 104 - scale_attention_scores (not yet solved)
# TODO: implement

# Step 105 - build_causal_mask (not yet solved)
# TODO: implement

# Step 106 - apply_causal_mask (not yet solved)
# TODO: implement

# Step 107 - softmax_attention_weights (not yet solved)
# TODO: implement

# Step 108 - attention_weighted_values (not yet solved)
# TODO: implement

# Step 109 - apply_output_projection (not yet solved)
# TODO: implement

# Step 110 - output_projection_backward (not yet solved)
# TODO: implement

# Step 111 - attention_value_backward (not yet solved)
# TODO: implement

# Step 112 - masked_softmax_backward (not yet solved)
# TODO: implement

# Step 113 - scale_scores_backward (not yet solved)
# TODO: implement

# Step 114 - qk_scores_backward (not yet solved)
# TODO: implement

# Step 115 - qkv_projection_backward (not yet solved)
# TODO: implement

# Step 116 - choose_attention_head_config (not yet solved)
# TODO: implement

# Step 117 - create_multihead_qkv_projections (not yet solved)
# TODO: implement

# Step 118 - create_multihead_output_projection (not yet solved)
# TODO: implement

# Step 119 - reshape_to_heads (not yet solved)
# TODO: implement

# Step 120 - transpose_heads_to_front (not yet solved)
# TODO: implement

# Step 121 - get_multihead_n_heads (not yet solved)
# TODO: implement

# Step 122 - get_multihead_sequence_length (not yet solved)
# TODO: implement

# Step 123 - compute_d_head (not yet solved)
# TODO: implement

# Step 124 - multihead_masked_softmax_scores (not yet solved)
# TODO: implement

# Step 125 - multihead_weighted_sum (not yet solved)
# TODO: implement

# Step 126 - transpose_heads_to_back (not yet solved)
# TODO: implement

# Step 127 - get_multihead_output_sequence_length (not yet solved)
# TODO: implement

# Step 128 - merge_heads_to_d_model (not yet solved)
# TODO: implement

# Step 129 - multihead_output_projection_forward (not yet solved)
# TODO: implement

# Step 130 - multihead_reshape_transpose_backward (not yet solved)
# TODO: implement

# Step 131 - ffn_linear_one_forward (not yet solved)
# TODO: implement

# Step 132 - ffn_activation_forward (not yet solved)
# TODO: implement

# Step 133 - ffn_linear_two_forward (not yet solved)
# TODO: implement

# Step 134 - ffn_backward (not yet solved)
# TODO: implement

# Step 135 - residual_forward (not yet solved)
# TODO: implement

# Step 136 - residual_backward (not yet solved)
# TODO: implement

# Step 137 - pre_layernorm_sublayer_forward (not yet solved)
# TODO: implement

# Step 138 - transformer_block_forward (not yet solved)
# TODO: implement

# Step 139 - transformer_block_backward (not yet solved)
# TODO: implement

# Step 140 - stack_transformer_blocks (not yet solved)
# TODO: implement

# Step 141 - forward_through_all_blocks (not yet solved)
# TODO: implement

# Step 142 - backward_through_all_blocks (not yet solved)
# TODO: implement

# Step 143 - final_layernorm_forward (not yet solved)
# TODO: implement

# Step 144 - lm_head_linear_forward (not yet solved)
# TODO: implement

# Step 145 - full_model_forward (not yet solved)
# TODO: implement

# Step 146 - full_model_backward (not yet solved)
# TODO: implement

# Step 147 - initialize_adam_moments (not yet solved)
# TODO: implement

# Step 148 - initialize_adam_step_counter (not yet solved)
# TODO: implement

# Step 149 - adam_increment_step (not yet solved)
# TODO: implement

# Step 150 - adam_update_first_moment (not yet solved)
# TODO: implement

# Step 151 - adam_update_second_moment (not yet solved)
# TODO: implement

# Step 152 - adam_bias_correction (not yet solved)
# TODO: implement

# Step 153 - adam_parameter_update (not yet solved)
# TODO: implement

# Step 154 - wire_full_training_loop (not yet solved)
# TODO: implement

# Step 155 - logging_and_validation_loss (not yet solved)
# TODO: implement

# Step 156 - encode_prompt (not yet solved)
# TODO: implement

# Step 157 - crop_context_to_block_size (not yet solved)
# TODO: implement

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

