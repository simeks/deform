#pragma once

#include "sub_function.h"

template<typename T>
struct SquaredDistanceFunction : public SubFunction
{
    SquaredDistanceFunction(const stk::VolumeHelper<T>& fixed,
                            const stk::VolumeHelper<T>& moving) :
        _fixed(fixed),
        _moving(moving)
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

        T moving_v = _moving.linear_at(moving_p, stk::Border_Constant);

        // [Filip]: Addition for partial-body registrations
        if (moving_p.x < 0 || moving_p.x >= _moving.size().x ||
            moving_p.y < 0 || moving_p.y >= _moving.size().y ||
            moving_p.z < 0 || moving_p.z >= _moving.size().z) {
            return 0;
        }


        // TODO: Float cast
        float f = float(_fixed(p) - moving_v);
        return f*f;
    }

    stk::VolumeHelper<T> _fixed;
    stk::VolumeHelper<T> _moving;
};

