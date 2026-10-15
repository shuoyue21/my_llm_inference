#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
question="${1:-请用一句话解释什么是大模型推理。}"
if [ "$#" -gt 0 ]; then shift; fi
printf -v prompt '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n' "$question"
exec ./build-qwen2/demo/llama_infer models/qwen2.5-0.5b-instruct/model.bin models/qwen2.5-0.5b-instruct/tokenizer.json --device "${QWEN_DEVICE:-cuda}" --prompt "$prompt" --max-new-tokens 128 "$@"
