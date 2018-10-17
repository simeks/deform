#pragma once

#include <stk/filters/sobel.h>
#include <stk/filters/gaussian_filter.h>

#include "sub_function.h"

template<typename T, bool use_mask>
struct GradientSSDFunction : public SubFunction
{
    GradientSSDFunction(const stk::VolumeHelper<T>& fixed,
                        const stk::VolumeHelper<T>& moving,
                        const float sigma) :
        _fixed(stk::sobel(stk::gaussian_filter_3d(fixed, sigma))),
        _moving(stk::sobel(stk::gaussian_filter_3d(moving, sigma)))
    {}

    float cost(const int3& p, const float3& def)
    {
        // [fixed] -> [world] -> [moving]
        const auto moving_p = _moving.point2index(_fixed.index2point(p) + def);

        // Check whether the point is masked out
        float mask_value = 1.0f;
        if constexpr (use_mask) {
            mask_value = _moving_mask.linear_at(moving_p, stk::Border_Constant);
            if (mask_value <= std::numeric_limits<float>::epsilon()) {
                return 0.0f;
            }
        }

        // [Filip]: Addition for partial-body registrations
        if (moving_p.x < 0 || moving_p.x >= _moving.size().x ||
            moving_p.y < 0 || moving_p.y >= _moving.size().y ||
            moving_p.z < 0 || moving_p.z >= _moving.size().z) {
            return 0;
        }

        float3 moving_v = _moving.linear_at(moving_p, stk::Border_Constant);

        return mask_value * stk::norm2(_fixed(p) - moving_v);
    }

    stk::VolumeHelper<float3> _fixed;
    stk::VolumeHelper<float3> _moving;
};

