#pragma once

#include "sub_function.h"


struct LandmarksFunction : public SubFunction
{
    LandmarksFunction(const std::vector<float3>& fixed_landmarks,
                      const std::vector<float3>& moving_landmarks,
                      const float3& fixed_origin,
                      const float3& fixed_spacing,
                      const dim3& fixed_size) :
        _landmarks {fixed_landmarks},
        _fixed_origin {fixed_origin},
        _fixed_spacing {fixed_spacing},
        _fixed_size {fixed_size}
    {
        ASSERT(fixed_landmarks.size() == moving_landmarks.size());
        for (size_t i = 0; i < fixed_landmarks.size(); ++i) {
            _displacements.push_back(moving_landmarks[i] - fixed_landmarks[i]);
        }
    }

    float cost(const int3& p, const float3& def)
    {
        float cost = 0.0f;
        const float epsilon = 1e-6f;

        const float3 fixed_p{
            static_cast<float>(p.x),
            static_cast<float>(p.y),
            static_cast<float>(p.z)
        };

        const float3 world_p = _fixed_origin + fixed_p * _fixed_spacing;

        for (size_t i = 0; i < _landmarks.size(); ++i) {
            cost += stk::norm2(def - _displacements[i]) /
                    (stk::norm2(_landmarks[i] - world_p) + epsilon);
        }

        return cost;
    }

    const std::vector<float3> _landmarks;
    std::vector<float3> _displacements;
    const float3 _fixed_origin;
    const float3 _fixed_spacing;
    const dim3 _fixed_size;
};

