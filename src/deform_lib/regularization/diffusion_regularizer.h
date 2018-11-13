#pragma once

#include "regularizer.h"

#include <vector>

struct DiffusionRegularizer : public Regularizer
{
    DiffusionRegularizer(
            const float weight,
            const float scale,
            float exponent,
            const float3& spacing,
            const stk::VolumeFloat3& initial)
        : Regularizer(weight, scale, exponent, spacing, initial)
    {
    }

    virtual ~DiffusionRegularizer() {}

    virtual double operator()(
            const int3& p,
            const int3& step,
            const float3& def0,
            const float3& def1,
            const float3& = {})
    {
        float3 step_in_mm {
            step.x*_spacing.x,
            step.y*_spacing.y,
            step.z*_spacing.z
        };

        // The diff should be relative to the initial displacement diff
        float3 diff = (def0-_initial(p)) - (def1-_initial(p+step));

        float dist_squared = stk::norm2(diff);
        float step_squared = _scale * stk::norm2(step_in_mm);

        float w = _weight;

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        /*
            Tissue-specific regularization
            Per edge weight: Mean term of neighboring voxels

            w = 0.5f*(weights(p) + weights(p+step))
        */

        if (_weight_map.valid())
            w = 0.5f*(_weight_map(p) + _weight_map(p+step));
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

        return w * std::pow(dist_squared / step_squared, _half_exponent);
    }
};

