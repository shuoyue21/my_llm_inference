# Upstream provenance

Reference: https://github.com/zjhellofss/KuiperLLama

Pinned commit: `83030c894da046902c8134110eb70ed59cb49486`. Verified against a fresh `git clone --depth 1` of the reference repository. The local checkout has the same commit. Uncommitted changes in the local reference checkout were not imported.

The pre-existing project already derives from KuiperLLama. This integration preserves its Tensor/Buffer/Layer implementation and imports the missing model, attention, sampler, and exporter code listed below. Local corrections and new runtime/tests are tracked in this repository. No LICENSE file was present at this upstream revision; this note does not grant a license.

- `hf_infer/llama3_infer.py`
- `hf_infer/qwen2_infer.py`
- `include/model/config.h`
- `include/model/llama3.h`
- `include/model/model.h`
- `include/model/qwen2.h`
- `include/model/qwen3.h`
- `include/model/raw_model_data.h`
- `include/sampler/argmax_sampler.h`
- `include/sampler/sampler.h`
- `source/model/llama3.cpp`
- `source/model/model.cpp`
- `source/model/qwen2.cpp`
- `source/model/qwen3.cpp`
- `source/model/raw_model_data.cpp`
- `source/op/kernels/cpu/mha_kernel.cpp`
- `source/op/kernels/cpu/mha_kernel.h`
- `source/op/kernels/cuda/argmax_kernel.cu`
- `source/op/kernels/cuda/argmax_kernel.cuh`
- `source/op/kernels/cuda/mha_kernel.cu`
- `source/op/kernels/cuda/mha_kernel.cuh`
- `source/op/mha.cpp`
- `source/sampler/argmax_sampler.cpp`
- `tools/config.json`
- `tools/export_llama.py`
- `tools/export_llama3.py`
- `tools/export_qwen2.py`
- `tools/export_qwen3/config.py`
- `tools/export_qwen3/load.py`
- `tools/export_qwen3/model.py`
- `tools/export_qwen3/write_bin.py`
- `tools/model.py`
- `tools/model_qwen2.py`
