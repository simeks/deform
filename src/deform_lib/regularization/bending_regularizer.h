#pragma once

#include "regularizer.h"

#include <vector>

struct BendingRegularizer : public Regularizer
{
    /*!
     * \brief Approximate the bending energy as the sum of the
     *        non-mixed second derivatives.
     */
    BendingRegularizer(
            const float weight,
            const float scale,
            float exponent,
            const float3& spacing,
            const stk::VolumeFloat3& initial)
        : Regularizer(weight, scale, exponent, spacing, initial)
    {
    }

    virtual ~BendingRegularizer() {}

    /*!
     * @param p    Current voxel coordinates.
     * @param step Distance to the neighbours.
     * @param def0 Displacement at the previous voxel.
     * @param def1 Displacement at the current voxel.
     * @param def2 Displacement at the next voxel.
     */
    virtual double operator()(
            const int3& p,
            const int3& step,
            const float3& def0,
            const float3& def1,
            const float3& def2 = {}) const
    {
        float3 step_in_mm {
            step.x*_spacing.x,
            step.y*_spacing.y,
            step.z*_spacing.z
        };

        // The diff should be relative to the initial displacement diff
        float3 diff = -2.0f * (def1 - _initial(p))
                      + (def0 - _initial(p - step))
                      + (def2 - _initial(p + step));

        // NOTE: the denominator of the central difference approximation of
        // the second derivative should be squared, but here it is not done
        // for numerical stability.
        float dist_squared = stk::norm2(diff);
        float step_squared = stk::norm2(step_in_mm);

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

        return w * std::pow(_scale * dist_squared / step_squared, _half_exponent);
    }

    virtual Settings::Regularizer type() const {
        return Settings::Regularizer::Bending;
    }
};

