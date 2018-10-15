#pragma once

#include "sub_function.h"

struct GpuCostFunction_SSD : public GpuSubFunction
{
    GpuCostFunction_SSD(const stk::GpuVolume& fixed,
                        const stk::GpuVolume& moving,
                        const stk::GpuVolume& fixed_mask,
                        const stk::GpuVolume& moving_mask
                        ) :
        _fixed(fixed),
        _moving(moving),
        _fixed_mask(fixed_mask),
        _moving_mask(moving_mask)
    {
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
    stk::GpuVolume _fixed_mask;
    stk::GpuVolume _moving_mask;
};

