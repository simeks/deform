#pragma once

#include "../transform.h"

#if DF_USE_CUDA
namespace stk {
    class GpuVolume;
}

namespace gpu {
    stk::GpuVolume transform_volume(
        const stk::GpuVolume& src, 
        const stk::GpuVolume& def, 
        transform::Interp i = transform::Interp_Linear
    );
}
#endif // DF_USE_CUDA
