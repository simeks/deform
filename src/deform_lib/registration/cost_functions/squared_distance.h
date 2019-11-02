#pragma once

#include "sub_function.h"

template<typename T>
struct SquaredDistanceFunction : public SubFunction
{
    SquaredDistanceFunction(const stk::VolumeHelper<T>& fixed,
                            const stk::VolumeHelper<T>& moving
                            ) :
        _fixed(fixed),
        _moving(moving)
    {}

    virtual ~SquaredDistanceFunction() {}

    float cost(const int3& p, const float3& def)
    {
        // [fixed] -> [world] -> [moving]
        const auto moving_p = _moving.point2index(_fixed.index2point(p) + def);

        // Check whether the point is masked out
        float mask_value = 1.0f;
        if (_moving_mask.valid()) {
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

        T moving_v = _moving.linear_at(moving_p, stk::Border_Constant);

        float f = float(_fixed(p) - moving_v);
        return mask_value * f*f;
    }

    stk::VolumeHelper<T> _fixed;
    stk::VolumeHelper<T> _moving;
};

