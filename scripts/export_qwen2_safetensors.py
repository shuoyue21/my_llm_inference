#!/usr/bin/env python3
"""Export unsharded HF Qwen2(.5) safetensors to the runtime's FP32 legacy format.
Only numpy is required. No model instantiation or transformers installation.
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model_dir", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--context", type=int, default=2048)
args = parser.parse_args()
config = json.loads((args.model_dir / "config.json").read_text())
if config.get("model_type") != "qwen2": raise ValueError("Only Qwen2/Qwen2.5 checkpoints are supported")
if config.get("rope_theta", 1000000) != 1000000 or config.get("rope_scaling"):
    raise ValueError("This runtime requires unscaled Qwen2 RoPE with theta=1000000")
if abs(config.get("rms_norm_eps", 1e-6) - 1e-6) > 1e-12: raise ValueError("Unsupported RMSNorm epsilon")
if not 0 < args.context <= config["max_position_embeddings"]: raise ValueError("Invalid context length")
checkpoint = args.model_dir / "model.safetensors"
with checkpoint.open("rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    metadata = json.loads(f.read(header_size))
data = np.memmap(checkpoint, mode="r", dtype=np.uint8, offset=8 + header_size)
def write_tensor(out, name):
    info = metadata[name]
    start, end = info["data_offsets"]
    raw = data[start:end]
    if info["dtype"] == "BF16": values = (raw.view("<u2").astype("<u4") << 16).view("<f4")
    elif info["dtype"] == "F16": values = raw.view("<f2").astype("<f4")
    elif info["dtype"] == "F32": values = raw.view("<f4")
    else: raise ValueError(f"Unsupported dtype {info['dtype']}")
    assert values.size == int(np.prod(info["shape"]))
    values.tofile(out)
layers = config["num_hidden_layers"]
dim = config["hidden_size"]
heads = config["num_attention_heads"]
kv_heads = config["num_key_value_heads"]
shared = config.get("tie_word_embeddings", False)
with args.output.open("wb") as out:
    out.write(struct.pack("<7i", dim, config["intermediate_size"], layers, heads, kv_heads,
                          config["vocab_size"] * (1 if shared else -1), args.context))
    write_tensor(out, "model.embed_tokens.weight")
    def write_layers(suffix):
        for layer in range(layers): write_tensor(out, f"model.layers.{layer}.{suffix}")
    write_layers("input_layernorm.weight")
    for projection in ["q", "k", "v"]:
        for layer in range(layers):
            for kind in ["weight", "bias"]:
                write_tensor(out, f"model.layers.{layer}.self_attn.{projection}_proj.{kind}")
    write_layers("self_attn.o_proj.weight")
    write_layers("post_attention_layernorm.weight")
    for projection in ["gate", "down", "up"]: write_layers(f"mlp.{projection}_proj.weight")
    write_tensor(out, "model.norm.weight")
    # Legacy format reserves cosine/sine tables. Runtime computes its own tables.
    np.zeros(args.context * (dim // heads), dtype="<f4").tofile(out)
    if not shared: write_tensor(out, "lm_head.weight")
print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                  "layers": layers, "context": args.context, "shared_classifier": shared}))
