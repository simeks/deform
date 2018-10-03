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
        // [fixed] -> [world] -> [moving]
        const auto moving_p = _moving.point2index(_fixed.index2point(p) + def);

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

