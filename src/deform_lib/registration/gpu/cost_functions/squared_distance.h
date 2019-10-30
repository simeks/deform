#pragma once

#include "cost_function.h"

struct GpuCostFunction_SSD : public GpuCostFunction
{
    GpuCostFunction_SSD(const stk::GpuVolume& fixed, const stk::GpuVolume& moving) :
        _fixed(fixed),
        _moving(moving)
    {
        ASSERT(_fixed.usage() == stk::gpu::Usage_PitchedPointer);
        ASSERT(_moving.usage() == stk::gpu::Usage_PitchedPointer);
    }
    virtual ~GpuCostFunction_SSD() {}

    void cost(
        GpuDisplacementField& df,
        const float3& delta,
        float weight,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cost_acc,
        stk::cuda::Stream& stream
    );

    stk::GpuVolume _fixed;
    stk::GpuVolume _moving;
};

