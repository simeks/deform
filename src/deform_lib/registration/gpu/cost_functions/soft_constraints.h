#pragma once

#include "cost_function.h"

struct GpuCostFunction_SoftConstraints : public GpuCostFunction
{
    GpuCostFunction_SoftConstraints(
        const stk::GpuVolume& constraint_mask,
        const stk::GpuVolume& constraint_values) :
        _constraint_mask(constraint_mask),
        _constraint_values(constraint_values)
    {
        ASSERT(_constraint_mask.usage() == stk::gpu::Usage_PitchedPointer);
        ASSERT(_constraint_values.usage() == stk::gpu::Usage_PitchedPointer);
    }
    virtual ~GpuCostFunction_SoftConstraints() {}

    void cost(
        GpuDisplacementField& df,
        const float3& delta,
        float weight,
        const int3& offset,
        const int3& dims,
        Settings::UpdateRule update_rule,
        stk::GpuVolume& cost_acc,
        stk::cuda::Stream& stream
    );

    stk::GpuVolume _constraint_mask;
    stk::GpuVolume _constraint_values;
};

