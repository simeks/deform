#pragma once

#include "../../config.h"

#ifdef DF_USE_CUDA

namespace stk
{
    class GpuVolume;
}

struct dim3;

namespace filters
{
namespace gpu
{
    /// Applies a gaussian prefilter and downsamples the volume by 2.
    stk::GpuVolume downsample_volume_by_2(const stk::GpuVolume& vol);

    /// Downsamples a 3d vectorfield and keeps the residual
    stk::GpuVolume downsample_vectorfield_by_2(const stk::GpuVolume& vol
#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
        , stk::GpuVolume& residual
#endif
    );

    /// Upsamples a 3d vectorfield and optionally applies the given residual
    /// residual : Passing an invalid GpuVolume (simply GpuVolume()) will behave the same as having a zero residual
    stk::GpuVolume upsample_vectorfield(const stk::GpuVolume& vol, const dim3& new_dims
#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    , const stk::GpuVolume& residual
#endif
    );
}
}
#endif
