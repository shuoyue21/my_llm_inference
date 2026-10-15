#!/usr/bin/env python3
"""Compare every prefill/decode logit against an independent PyTorch FP32 Llama forward.
Requires numpy and torch. The engine itself does not depend on Python/PyTorch.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy as np
import torch


def reference(checkpoint, tokens):
    with open(checkpoint, "rb") as f:
        d, hidden, layers, heads, kv_heads, vocab_signed, context = struct.unpack("7i", f.read(28))
    vocab = abs(vocab_signed)
    hs, kv = d // heads, d * kv_heads // heads
    weights = np.memmap(checkpoint, dtype="<f4", mode="r", offset=28)
    offset = 0
    def take(*shape):
        nonlocal offset
        size = int(np.prod(shape))
        result = torch.from_numpy(np.array(weights[offset:offset + size]).reshape(shape))
        offset += size
        return result
    embedding = take(vocab, d)
    attn_norm = take(layers, d)
    wq = take(layers, d, d)
    wk = take(layers, kv, d)
    wv = take(layers, kv, d)
    wo = take(layers, d, d)
    ffn_norm = take(layers, d)
    w1 = take(layers, hidden, d)
    w2 = take(layers, d, hidden)
    w3 = take(layers, hidden, d)
    final_norm = take(d)
    classifier = embedding if vocab_signed > 0 else torch.from_numpy(
        np.array(weights[offset + context * hs:offset + context * hs + vocab * d]).reshape(vocab, d))
    keys = torch.empty(layers, len(tokens), kv_heads, hs)
    values = torch.empty_like(keys)
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, hs, 2).float() / hs))
    def norm(x, w): return x * torch.rsqrt(x.square().mean() + 1e-5) * w
    def rotate(x, position):
        pairs = x.reshape(-1, hs // 2, 2)
        cos, sin = torch.cos(position * frequencies), torch.sin(position * frequencies)
        return torch.stack((pairs[..., 0] * cos - pairs[..., 1] * sin,
                            pairs[..., 0] * sin + pairs[..., 1] * cos), dim=-1).flatten(-2)
    for position, token in enumerate(tokens):
        x = embedding[token].clone()
        for layer in range(layers):
            normalized = norm(x, attn_norm[layer])
            q = rotate(wq[layer] @ normalized, position)
            keys[layer, position] = rotate(wk[layer] @ normalized, position)
            values[layer, position] = (wv[layer] @ normalized).reshape(kv_heads, hs)
            k = keys[layer, :position + 1].repeat_interleave(heads // kv_heads, dim=1)
            v = values[layer, :position + 1].repeat_interleave(heads // kv_heads, dim=1)
            scores = torch.einsum("hd,thd->ht", q, k) / (hs ** 0.5)
            attention = torch.einsum("ht,thd->hd", scores.softmax(-1), v).flatten()
            x = x + wo[layer] @ attention
            normalized = norm(x, ffn_norm[layer])
            x = x + w2[layer] @ (torch.nn.functional.silu(w1[layer] @ normalized) * (w3[layer] @ normalized))
        yield (classifier @ norm(x, final_norm)).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/demo/llama_infer")
    parser.add_argument("--model", default="models/stories15M.bin")
    parser.add_argument("--tokenizer", default="models/tokenizer.model")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory() as directory:
        dump = Path(directory) / "logits.f32"
        command = [args.binary, args.model, args.tokenizer, "--device", args.device,
                   "--prompt", args.prompt, "--max-new-tokens", str(args.tokens), "--json",
                   "--dump-logits", str(dump)]
        completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        result = json.loads(completed.stdout)
        vocab = abs(struct.unpack("7i", Path(args.model).read_bytes()[:28])[5])
        actual = np.fromfile(dump, dtype="<f4").reshape(-1, vocab)
        prompt = result["prompt_tokens"]
        generated = result["generated_tokens"]
        input_ids = prompt + generated[:-1]
        assert len(actual) == len(input_ids), (actual.shape, len(input_ids))
        worst = 0.0
        for position, expected in enumerate(reference(args.model, input_ids)):
            np.testing.assert_allclose(actual[position], expected, atol=3e-4, rtol=3e-4,
                                       err_msg=f"{args.device} position {position}")
            worst = max(worst, float(np.max(np.abs(actual[position] - expected))))
            if position >= len(prompt) - 1:
                assert int(expected.argmax()) == generated[position - len(prompt) + 1]
        print(json.dumps({"device": args.device, "positions": len(actual), "vocab_size": vocab,
                          "max_abs_logit_error": worst, "generated_tokens_match": True,
                          "text": result["text"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
