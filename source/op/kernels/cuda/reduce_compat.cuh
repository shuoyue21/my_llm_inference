#pragma once
#include <cub/version.cuh>
#if CUB_VERSION >= 300000
#include <cuda/functional>
#else
#include <cub/thread/thread_operators.cuh>
#endif

namespace kernel {
// Use each CUB version's native maximum operator so its reduction dispatch
// remains intact. CUB 3 replaced cub::Max with cuda::maximum.
#if CUB_VERSION >= 300000
using FloatMaximum = cuda::maximum<float>;
#else
using FloatMaximum = cub::Max;
#endif
}  // namespace kernel
