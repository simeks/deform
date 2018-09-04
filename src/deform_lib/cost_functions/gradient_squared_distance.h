#pragma once

#include <stk/filters/sobel.h>
#include <stk/filters/gaussian_filter.h>

#include "sub_function.h"

template<typename T>
struct GradientSSDFunction : public SubFunction
{
    GradientSSDFunction(const stk::VolumeHelper<T>& fixed,
                        const stk::VolumeHelper<T>& moving,
                        const int sigma) :
        _fixed(stk::sobel(stk::gaussian_filter_3d(fixed, sigma))),
        _moving(stk::sobel(stk::gaussian_filter_3d(moving, sigma)))
    {}

    float cost(const int3& p, const float3& def)
    {
        float3 fixed_p{
            float(p.x),
            float(p.y),
            float(p.z)
        };
        
        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p + def - _moving.origin()) / _moving.spacing();

        float3 moving_v = _moving.linear_at(moving_p, stk::Border_Constant);

        // [Filip]: Addition for partial-body registrations
        if (moving_p.x < 0 || moving_p.x >= _moving.size().x ||
            moving_p.y < 0 || moving_p.y >= _moving.size().y ||
            moving_p.z < 0 || moving_p.z >= _moving.size().z) {
            return 0;
        }


        return stk::norm2(_fixed(p) - moving_v);
    }

    stk::VolumeHelper<float3> _fixed;
    stk::VolumeHelper<float3> _moving;
};

