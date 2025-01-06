{
  "_name_or_path": "smol-8x101-ft",
  "architectures": [
    "MixtralForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 32768,
  "model_type": "mixtral",
  "num_attention_heads": 24,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 6,
  "num_key_value_heads": 8,
  "num_local_experts": 8,
  "output_router_logits": true,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "router_aux_loss_coef": 0.001,
  "sliding_window": 1024,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.37.0.dev0",
  "use_cache": false,
  "vocab_size": 32128
}




MixtralForCausalLM(
  (model): MixtralModel(
    (embed_tokens): Embedding(32128, 768)
    (layers): ModuleList(
      (0-5): 6 x MixtralDecoderLayer(
        (self_attn): MixtralSdpaAttention(
          (q_proj): Linear(in_features=768, out_features=768, bias=False)
          (k_proj): Linear(in_features=768, out_features=256, bias=False)
          (v_proj): Linear(in_features=768, out_features=256, bias=False)
          (o_proj): Linear(in_features=768, out_features=768, bias=False)
          (rotary_emb): MixtralRotaryEmbedding()
        )
        (block_sparse_moe): Merge_MixtralSparseMoeBlock(
          (gate): Linear(in_features=768, out_features=8, bias=False)
          (Wmean1): Linear(in_features=768, out_features=3072, bias=False)
          (Wmean2): Linear(in_features=3072, out_features=768, bias=False)
          (Wmean3): Linear(in_features=768, out_features=3072, bias=False)
          (experts_delta_v1_shared): Linear(in_features=768, out_features=307, bias=False)
          (experts_delta_v2_shared): Linear(in_features=3072, out_features=307, bias=False)
          (experts_delta_v3_shared): Linear(in_features=768, out_features=307, bias=False)
          (experts): ModuleList(
            (0-7): 8 x meanW_deltaUV(
              (Wmean1): Linear(in_features=768, out_features=3072, bias=False)
              (Wmean2): Linear(in_features=3072, out_features=768, bias=False)
              (Wmean3): Linear(in_features=768, out_features=3072, bias=False)
              (act_fn): SiLU()
              (delta_u1): Linear(in_features=307, out_features=3072, bias=False)
              (delta_v1): Linear(in_features=768, out_features=307, bias=False)
              (delta_u2): Linear(in_features=307, out_features=768, bias=False)
              (delta_v2): Linear(in_features=3072, out_features=307, bias=False)
              (delta_u3): Linear(in_features=307, out_features=3072, bias=False)
              (delta_v3): Linear(in_features=768, out_features=307, bias=False)
            )
          )
        )
        (input_layernorm): MixtralRMSNorm()
        (post_attention_layernorm): MixtralRMSNorm()
      )
    )
    (norm): MixtralRMSNorm()
  )
  (lm_head): Linear(in_features=768, out_features=32128, bias=False)
)









MixtralForCausalLM(
  (model): MixtralModel(
    (embed_tokens): Embedding(32128, 768)
    (layers): ModuleList(
      (0-5): 6 x MixtralDecoderLayer(
        (self_attn): MixtralSdpaAttention(
          (q_proj): Linear(in_features=768, out_features=768, bias=False)
          (k_proj): Linear(in_features=768, out_features=256, bias=False)
          (v_proj): Linear(in_features=768, out_features=256, bias=False)
          (o_proj): Linear(in_features=768, out_features=768, bias=False)
          (rotary_emb): MixtralRotaryEmbedding()
        )
        (block_sparse_moe): MixtralSparseMoeBlock(
          (gate): Linear(in_features=768, out_features=8, bias=False)
          (experts): ModuleList(
            (0-7): 8 x MixtralBlockSparseTop2MLP(
              (w1): Linear(in_features=768, out_features=3072, bias=False)
              (w2): Linear(in_features=3072, out_features=768, bias=False)
              (w3): Linear(in_features=768, out_features=3072, bias=False)
              (act_fn): SiLU()
            )
          )
        )
        (input_layernorm): MixtralRMSNorm()
        (post_attention_layernorm): MixtralRMSNorm()
      )
    )
    (norm): MixtralRMSNorm()
  )
  (lm_head): Linear(in_features=768, out_features=32128, bias=False)
)




---------------------------------------------------------------------------------------------
{
  "architectures": [
    "MixtralForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 32768,
  "model_type": "mixtral",
  "num_attention_heads": 32,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "num_local_experts": 8,
  "output_router_logits": false,
  "rms_norm_eps": 1e-05,
  "rope_theta": 1000000.0,
  "router_aux_loss_coef": 0.02,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.36.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}




model
MixtralForCausalLM(
  (model): MixtralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x MixtralDecoderLayer(
        (self_attn): MixtralSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MixtralRotaryEmbedding()
        )
        (block_sparse_moe): MixtralSparseMoeBlock(
          (gate): Linear(in_features=4096, out_features=8, bias=False)
          (experts): ModuleList(
            (0-7): 8 x MixtralBlockSparseTop2MLP(
              (w1): Linear(in_features=4096, out_features=14336, bias=False)
              (w2): Linear(in_features=14336, out_features=4096, bias=False)
              (w3): Linear(in_features=4096, out_features=14336, bias=False)
              (act_fn): SiLU()
            )
          )
        )
        (input_layernorm): MixtralRMSNorm()
        (post_attention_layernorm): MixtralRMSNorm()
      )
    )
    (norm): MixtralRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)

model
MixtralForCausalLM(
  (model): MixtralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x MixtralDecoderLayer(
        (self_attn): MixtralSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MixtralRotaryEmbedding()
        )
        (block_sparse_moe): Merge_MixtralSparseMoeBlock(
          (gate): Linear(in_features=4096, out_features=8, bias=False)
          (Wmean1): Linear(in_features=4096, out_features=14336, bias=False)
          (Wmean2): Linear(in_features=14336, out_features=4096, bias=False)
          (Wmean3): Linear(in_features=4096, out_features=14336, bias=False)
          (experts_delta_v1_shared): Linear(in_features=4096, out_features=1592, bias=False)
          (experts_delta_v2_shared): Linear(in_features=14336, out_features=1592, bias=False)
          (experts_delta_v3_shared): Linear(in_features=4096, out_features=1592, bias=False)
          (experts): ModuleList(
            (0-7): 8 x meanW_deltaUV(
              (Wmean1): Linear(in_features=4096, out_features=14336, bias=False)
              (Wmean2): Linear(in_features=14336, out_features=4096, bias=False)
              (Wmean3): Linear(in_features=4096, out_features=14336, bias=False)
              (act_fn): SiLU()
              (delta_u1): Linear(in_features=1592, out_features=14336, bias=False)
              (delta_v1): Linear(in_features=4096, out_features=1592, bias=False)
              (delta_u2): Linear(in_features=1592, out_features=4096, bias=False)
              (delta_v2): Linear(in_features=14336, out_features=1592, bias=False)
              (delta_u3): Linear(in_features=1592, out_features=14336, bias=False)
              (delta_v3): Linear(in_features=4096, out_features=1592, bias=False)
            )
          )
        )
        (input_layernorm): MixtralRMSNorm()
        (post_attention_layernorm): MixtralRMSNorm()
      )
    )
    (norm): MixtralRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
