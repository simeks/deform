#pragma once

#include "sub_function.h"

struct SoftConstraintsFunction : public SubFunction
{
    SoftConstraintsFunction(const stk::VolumeUChar& constraint_mask,
                            const stk::VolumeFloat3& constraints_values) :
        _constraints_mask(constraint_mask),
        _constraints_values(constraints_values),
        _spacing(_constraints_values.spacing())
    {}

    virtual ~SoftConstraintsFunction() {}

    float cost(const int3& p, const float3& def)
    {
        if (_constraints_mask(p) != 0)
        {
            float3 diff = def - _constraints_values(p);

            // Distance^2 in [mm]
            float dist_squared = stk::norm2(diff);

            return std::min(dist_squared, 1000.0f); // Clamp to avoid explosion
        }
        return 0.0f;
    }
    stk::VolumeUChar _constraints_mask;
    stk::VolumeFloat3 _constraints_values;
    float3 _spacing;
};

