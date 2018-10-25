#pragma once

#include <stk/common/assert.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/float4.h>
#include <stk/math/matrix3x3f.h>

namespace stk { namespace cuda {
    class Stream;
}}

struct GpuSubFunction
{
    GpuSubFunction() {}
    virtual ~GpuSubFunction() {}

    // Costs are accumulated into the specified cost_acc volume (of type float2),
    //  where x is the cost before applying delta and y is the cost after.
    // offset       : Offset to region to compute terms in
    // dims         : Size of region
    // df           : Displacement field
    // cost_acc     : Destination for cost (float2, with cost before (x) and after (y) applying delta)
    // delta        : Delta applied to the displacement, typically based on the step-size.
    virtual void cost(
        stk::GpuVolume& df,
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

