#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model_dir=models/qwen2.5-0.5b-instruct
revision=7ae557604adf67be50417f59c2c2f167def9a775
mkdir -p "$model_dir"
for file in config.json tokenizer.json model.safetensors; do
  curl -fL --retry 3 "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/$revision/$file" -o "$model_dir/$file"
done
python3 scripts/export_qwen2_safetensors.py "$model_dir" "$model_dir/model.bin" --context 2048
