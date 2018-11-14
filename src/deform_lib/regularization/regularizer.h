#pragma once

#include "../registration/settings.h"

#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>

struct Regularizer
{
    Regularizer(
            const float weight,
            const float scale,
            const float exponent,
            const float3& spacing,
            const stk::VolumeFloat3& initial)
        : _weight(weight)
        , _scale(scale)
        , _half_exponent(0.5f * exponent)
        , _spacing(spacing)
        , _initial(initial)
    {
    }

    virtual ~Regularizer() {}

    /// p   : Position in fixed image
    /// step : Direction to neighbor
    /// def0 : Deformation in active voxel [mm]
    /// def1 : Deformation in preceding neighbor [mm]
    /// def2 : Deformation in succeeding neighbor [mm]
    virtual double operator()(
            const int3& p,
            const int3& step,
            const float3& def0,
            const float3& def1,
            const float3& def2 = {}) const = 0;

    virtual Settings::Regularizer type() const = 0;

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_weight_map(stk::VolumeFloat& map) { _weight_map = map; }
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    const float _weight;
    const float _scale;
    const float _half_exponent;
    const float3 _spacing;

    // Initial displacement for this registration level. This will be
    //  the reference when computing the regularization energy. Any
    //  displacement identical to the initial displacement will result in
    //  zero energy.
    const stk::VolumeFloat3 _initial;

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    stk::VolumeFloat _weight_map;
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP
};
