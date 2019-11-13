#pragma once

#include "cost_function.h"

#include <deform_lib/registration/settings.h>

#include <stk/cuda/cuda.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>

#include <thrust/device_vector.h>

namespace cuda {
    using namespace stk::cuda;
}

struct GpuCostFunction_Landmarks : public GpuCostFunction
{
    GpuCostFunction_Landmarks(
        const stk::GpuVolume& fixed,
        const std::vector<float3>& fixed_landmarks,
        const std::vector<float3>& moving_landmarks,
        const float decay
    );
    ~GpuCostFunction_Landmarks();
    
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

    float3 _origin;
    float3 _spacing;
    Matrix3x3f _direction;

    thrust::device_vector<float4> _landmarks;
    thrust::device_vector<float4> _displacements;

    float _half_decay;
};