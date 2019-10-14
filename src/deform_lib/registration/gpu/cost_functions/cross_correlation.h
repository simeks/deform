#pragma once

#include "cost_function.h"

struct GpuCostFunction_NCC : public GpuCostFunction
{
    GpuCostFunction_NCC(const stk::GpuVolume& fixed,
                        const stk::GpuVolume& moving,
                        int radius) :
        _fixed(fixed),
        _moving(moving),
        _radius(radius)
    {
        ASSERT(_fixed.usage() == stk::gpu::Usage_PitchedPointer);
        ASSERT(_moving.usage() == stk::gpu::Usage_PitchedPointer);
    }
    virtual ~GpuCostFunction_NCC() {}

    void cost(
        stk::GpuVolume& df,
        const float3& delta,
        float weight,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cost_acc,
        Settings::UpdateRule update_rule,
        stk::cuda::Stream& stream
    );

    stk::GpuVolume _fixed;
    stk::GpuVolume _moving;
    int _radius;
};

