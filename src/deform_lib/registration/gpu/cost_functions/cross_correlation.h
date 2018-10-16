#pragma once

#include "sub_function.h"

struct GpuCostFunction_NCC : public GpuSubFunction
{
    GpuCostFunction_NCC(const stk::GpuVolume& fixed,
                        const stk::GpuVolume& moving,
                        const stk::GpuVolume& fixed_mask,
                        const stk::GpuVolume& moving_mask,
                        int radius) :
        _fixed(fixed),
        _moving(moving),
        _fixed_mask(fixed_mask),
        _moving_mask(moving_mask),
        _radius(radius)
    {
    }
    ~GpuCostFunction_NCC() {}

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
    const stk::GpuVolume _fixed_mask;
    const stk::GpuVolume _moving_mask;
    int _radius;
};

