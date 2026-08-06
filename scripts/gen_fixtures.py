"""Generate correctness fixtures for the Swift runtime.

Builds a tiny, seeded-random qwen3_next model with mlx-lm's reference
implementation, saves its weights + config, runs one teacher-forcing forward
pass, and dumps every layer's output hidden states plus final logits.

The Swift CPU reference (milestone M2) and Metal kernels (M3) must reproduce
these tensors. Regenerate with:  .venv/bin/python scripts/gen_fixtures.py
"""

import json
import pathlib

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_next

OUT = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
TINY = OUT / "tiny-model"
TINY.mkdir(parents=True, exist_ok=True)

ARGS = dict(
    model_type="qwen3_next",
    hidden_size=64,
    num_hidden_layers=8,          # 6 DeltaNet + 2 full attention (interval 4)
    intermediate_size=128,        # dense-MLP fallback, unused (sparse step 1)
    num_attention_heads=4,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    num_experts=8,
    num_experts_per_tok=2,
    decoder_sparse_step=1,
    shared_expert_intermediate_size=32,
    mlp_only_layers=[],
    moe_intermediate_size=32,
    rms_norm_eps=1e-6,
    vocab_size=128,
    num_key_value_heads=2,
    rope_theta=10_000_000.0,
    partial_rotary_factor=0.25,
    max_position_embeddings=512,
    head_dim=16,
    norm_topk_prob=True,
    tie_word_embeddings=False,
    full_attention_interval=4,
)

# >= 50 tokens: long enough for position-dependent bugs (RoPE drift, conv
# tail, delta-state decay) to surface — 10-token fixtures let the qwen3_5
# norm_topk_prob default bug through unseen.
TOKENS = [
    1, 5, 9, 42, 7, 99, 3, 17, 64, 2,
    23, 88, 101, 54, 12, 71, 33, 90, 6, 47,
    120, 15, 78, 29, 4, 61, 110, 8, 36, 95,
    50, 27, 83, 14, 68, 41, 126, 19, 73, 32,
    58, 11, 104, 45, 21, 87, 66, 30, 116, 53,
    25, 92, 38, 79, 16, 107,
]


def main() -> None:
    # CPU device: mlx's ops-path semantics (exact reference), not the fast
    # approximate GPU kernels. Keeps the oracle deterministic and comparable
    # to the Swift f32 implementation at ~1e-5.
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    model = qwen3_next.Model(qwen3_next.ModelArgs(**ARGS))
    mx.eval(model.parameters())

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(TINY / "model.safetensors"), weights)
    (TINY / "config.json").write_text(json.dumps(ARGS, indent=2))

    inputs = mx.array([TOKENS])
    inner = model.model

    fixtures: dict[str, mx.array] = {"input_ids": inputs}
    h = inner.embed_tokens(inputs)
    fixtures["embed"] = h

    # Mirror Qwen3NextModel.__call__ so each layer's output can be captured.
    cache = [None] * len(inner.layers)
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    fa_mask = create_attention_mask(h, cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
    for i, layer in enumerate(inner.layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=None)
        fixtures[f"layer_{i:02d}"] = h

    h = inner.norm(h)
    fixtures["final_norm"] = h
    logits = model.lm_head(h)
    fixtures["logits"] = logits
    mx.eval(fixtures)

    mx.save_safetensors(str(OUT / "tiny_forward.safetensors"), fixtures)

    greedy = mx.argmax(logits[0], axis=-1).tolist()
    manifest = {
        "tokens": TOKENS,
        "greedy_next_tokens": greedy,
        "tensors": sorted(fixtures.keys()),
        "dtype": str(logits.dtype),
        "mlx_seed": 0,
    }
    (OUT / "tiny_forward.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}/tiny_forward.safetensors ({len(fixtures)} tensors)")
    print(f"greedy next-token ids: {greedy}")

    # Quantized variant (MLX affine int4, group 32 to fit the tiny dims):
    # validates the Swift dequantization path end-to-end.
    import mlx.nn as nn

    mx.random.seed(0)
    qmodel = qwen3_next.Model(qwen3_next.ModelArgs(**ARGS))
    mx.eval(qmodel.parameters())
    nn.quantize(qmodel, group_size=32, bits=4)
    mx.eval(qmodel.parameters())

    qtiny = OUT / "tiny-model-q4"
    qtiny.mkdir(exist_ok=True)
    mx.save_safetensors(str(qtiny / "model.safetensors"), dict(tree_flatten(qmodel.parameters())))
    (qtiny / "config.json").write_text(
        json.dumps({**ARGS, "quantization": {"group_size": 32, "bits": 4}}, indent=2)
    )

    q_logits = qmodel(inputs)
    mx.eval(q_logits)
    mx.save_safetensors(
        str(OUT / "tiny_forward_q4.safetensors"),
        {"input_ids": inputs, "logits": q_logits},
    )
    q_greedy = mx.argmax(q_logits[0], axis=-1).tolist()
    (OUT / "tiny_forward_q4.json").write_text(
        json.dumps({"tokens": TOKENS, "greedy_next_tokens": q_greedy}, indent=2)
    )
    print(f"wrote quantized tiny model + fixture; greedy: {q_greedy}")

    gen_qwen3_5_fixtures()


def gen_qwen3_5_fixtures() -> None:
    """Tiny qwen3_5 (Qwen3.5/3.6 family) fixtures: split DeltaNet projections,
    weights saved under the language_model. prefix, nested text_config — the
    exact shape of real multimodal checkpoints."""
    import dataclasses

    from mlx_lm.models import qwen3_5
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    tiny = dict(
        model_type="qwen3_5",
        hidden_size=64,
        num_hidden_layers=8,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=32,
        moe_intermediate_size=32,
        rms_norm_eps=1e-6,
        vocab_size=128,
        max_position_embeddings=512,
        full_attention_interval=4,
        # norm_topk_prob deliberately OMITTED: real Qwen3.5/3.6 checkpoints
        # don't carry the key and rely on TextModelArgs' family default
        # (True). The Swift config parser must reproduce that default.
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    valid = {f.name for f in dataclasses.fields(qwen3_5.TextModelArgs)}
    args = qwen3_5.TextModelArgs.from_dict({k: v for k, v in tiny.items() if k in valid})

    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    model = qwen3_5.TextModel(args)
    mx.eval(model.parameters())

    q35 = OUT / "tiny-model-q35"
    q35.mkdir(exist_ok=True)
    weights = {"language_model." + k: v for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(q35 / "model.safetensors"), weights)
    (q35 / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "text_config": tiny}, indent=2)
    )

    inputs = mx.array([TOKENS])
    fixtures: dict[str, mx.array] = {"input_ids": inputs}
    inner = model.model
    h = inner.embed_tokens(inputs)
    fixtures["embed"] = h
    cache = [None] * len(inner.layers)
    fa_mask = create_attention_mask(h, cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
    for i, layer in enumerate(inner.layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=None)
        fixtures[f"layer_{i:02d}"] = h
    h = inner.norm(h)
    fixtures["final_norm"] = h
    logits = model.lm_head(h)
    fixtures["logits"] = logits
    mx.eval(fixtures)

    mx.save_safetensors(str(OUT / "tiny_forward_q35.safetensors"), fixtures)
    greedy = mx.argmax(logits[0], axis=-1).tolist()
    (OUT / "tiny_forward_q35.json").write_text(
        json.dumps({"tokens": TOKENS, "greedy_next_tokens": greedy}, indent=2)
    )
    print(f"wrote qwen3_5 tiny model + fixture; greedy: {greedy}")


if __name__ == "__main__":
    main()
