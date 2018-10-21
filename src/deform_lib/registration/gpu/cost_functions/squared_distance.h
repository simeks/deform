#pragma once

#include "sub_function.h"

struct GpuCostFunction_SSD : public GpuSubFunction
{
    GpuCostFunction_SSD(const stk::GpuVolume& fixed, const stk::GpuVolume& moving) :
        _fixed(fixed),
        _moving(moving)
    {
        ASSERT(_fixed.usage() == stk::gpu::Usage_PitchedPointer);
        ASSERT(_moving.usage() == stk::gpu::Usage_PitchedPointer);
    }
    ~GpuCostFunction_SSD() {}

    void cost(
        stk::GpuVolume& df,
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

