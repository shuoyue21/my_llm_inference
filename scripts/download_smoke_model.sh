#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
curl -fL --retry 3 https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories15M.bin -o models/stories15M.bin
curl -fL --retry 3 https://huggingface.co/yahma/llama-7b-hf/resolve/main/tokenizer.model -o models/tokenizer.model
printf '%s  %s\n' cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a models/stories15M.bin | sha256sum -c -
