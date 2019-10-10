#pragma once

#include "cost_function.h"

#include <deform_lib/registration/settings.h>

#include <stk/cuda/cuda.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>

#include <thrust/device_vector.h>

namespace cuda = stk::cuda;

struct GpuCostFunction_Landmarks : public GpuCostFunction
{
    GpuCostFunction_Landmarks(
        const stk::GpuVolume& fixed,
        const std::vector<float3>& fixed_landmarks,
        const std::vector<float3>& moving_landmarks,
        const float decay
    ) :
        _origin(fixed.origin()),
        _spacing(fixed.spacing()),
        _direction(fixed.direction()),
        _landmarks(fixed_landmarks),
        _half_decay(decay / 2.0f)
    {
        std::vector<float3> displacements;
        ASSERT(fixed_landmarks.size() == moving_landmarks.size());
        for (size_t i = 0; i < fixed_landmarks.size(); ++i) {
            displacements.push_back(moving_landmarks[i] - fixed_landmarks[i]);
        }
        _displacements = displacements;
    }

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

    float3 _origin;
    float3 _spacing;
    Matrix3x3f _direction;

    thrust::device_vector<float3> _landmarks;
    thrust::device_vector<float3> _displacements;

    float _half_decay;
};