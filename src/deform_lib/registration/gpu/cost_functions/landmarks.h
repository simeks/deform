#pragma once

#include "sub_function.h"

#include <thrust/device_vector.h>

struct GpuCostFunction_Landmarks : public GpuSubFunction
{
    GpuCostFunction_Landmarks(const std::vector<float3>& fixed_landmarks,
                              const std::vector<float3>& moving_landmarks,
                              const stk::GpuVolume& fixed,
                              const float decay);

    virtual ~GpuCostFunction_Landmarks();

    void cost(
        stk::GpuVolume& df,
        const float3& delta,
        float weight,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cap_source,
        stk::GpuVolume& cap_sink,
        stk::cuda::Stream& stream
    );

    const thrust::device_vector<float3> _landmarks;
    thrust::device_vector<float3> _displacements;
    const stk::GpuVolume _fixed;
    const float _half_decay;
};

