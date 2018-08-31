#pragma once

#ifdef DF_USE_CUDA
namespace stk
{
    class GpuVolume;
}

namespace filters {
namespace gpu {
    /// Gaussian filter for 3d volumes on GPU
    stk::GpuVolume gaussian_filter_3d(const stk::GpuVolume& volume, float sigma);
}
}
#endif