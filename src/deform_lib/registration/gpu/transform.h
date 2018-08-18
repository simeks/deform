#pragma once

#include "../transform.h"

#include <stk/image/dim3.h>

#if DF_USE_CUDA
namespace stk {
    class GpuVolume;
}

namespace gpu {
    stk::GpuVolume transform_volume(
        const stk::GpuVolume& src, 
        const stk::GpuVolume& def, 
        transform::Interp i = transform::Interp_Linear,
        const dim3& block_size = {32, 32, 1}
    );
}
#endif // DF_USE_CUDA
