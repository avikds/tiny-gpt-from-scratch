# Tiny GPT From Scratch

Build a minimal GPT-style language model end-to-end in pure NumPy, from character tokenization and softmax up through multi-head self-attention, transformer blocks, Adam, and sampling. Every forward and backward pass is derived and implemented by hand so you understand exactly how a transformer trains and generates text.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** build_char_vocab
- [x] **2.** build_id_to_char
- [x] **3.** encode_text
- [x] **4.** decode_ids
- [x] **5.** make_batches
- [x] **6.** token_embedding_lookup
- [x] **7.** add_positional_embeddings
- [x] **8.** linear_projection
- [x] **9.** compute_attention_scores
- [x] **10.** scale_attention_scores
- [x] **11.** apply_causal_mask
- [x] **12.** softmax_attention_weights
- [x] **13.** attention_context
- [x] **14.** split_heads
- [x] **15.** merge_heads
- [x] **16.** project_qkv
- [x] **17.** multi_head_scaled_dot_product_attention
- [x] **18.** merge_and_output_project
- [x] **19.** masked_multi_head_self_attention
- [x] **20.** gelu_activation
- [x] **21.** ffn_first_layer
- [x] **22.** ffn_second_layer
- [x] **23.** position_wise_feed_forward
- [x] **24.** layernorm_stats
- [x] **25.** layer_norm
- [x] **26.** pre_norm_residual_sublayer
- [x] **27.** transformer_block
- [x] **28.** gpt_backbone
- [x] **29.** project_to_vocab_logits
- [x] **30.** gpt_forward
- [x] **31.** cross_entropy_language_modeling_loss
- [x] **32.** init_gpt_parameters
- [x] **33.** collect_parameters
- [x] **34.** training_step
- [x] **35.** apply_optimizer_update
- [x] **36.** run_training_loop
- [ ] **37.** last_position_logits
- [ ] **38.** scale_logits_by_temperature
- [ ] **39.** top_k_filter_logits
- [ ] **40.** sample_next_token
- [ ] **41.** generate_text

---

Built on Deep-ML.
