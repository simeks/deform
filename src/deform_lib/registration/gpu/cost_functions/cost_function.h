#pragma once

#include <deform_lib/registration/settings.h>

#include <stk/cuda/cuda.h>
#include <stk/image/gpu_volume.h>

#include <thrust/device_vector.h>

namespace cuda {
    using namespace stk::cuda;
}


class GpuDisplacementField;

struct GpuCostFunction
{
    GpuCostFunction() {}
    virtual ~GpuCostFunction() {}

    // Costs are accumulated into the specified cost_acc volume (of type float2),
    //  where x is the cost before applying delta and y is the cost after.
    // df           : Displacement field
    // delta        : Delta applied to the displacement, typically based on the step-size.
    // offset       : Offset to region to compute terms in
    // dims         : Size of region
    // cost_acc     : Destination for cost (float2, with cost before (x) and after (y) applying delta)
    virtual void cost(
        GpuDisplacementField& df,
        const float3& delta,
        float weight,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cost_acc,
        stk::cuda::Stream& stream
    ) = 0;

    void set_fixed_mask(const stk::GpuVolume& fixed_mask)
    {
        ASSERT(fixed_mask.usage() == stk::gpu::Usage_PitchedPointer);
        _fixed_mask = fixed_mask;
    }

    void set_moving_mask(const stk::GpuVolume& moving_mask)
    {
        ASSERT(moving_mask.usage() == stk::gpu::Usage_PitchedPointer);
        _moving_mask = moving_mask;
    }

    stk::GpuVolume _fixed_mask;
    stk::GpuVolume _moving_mask;
};

