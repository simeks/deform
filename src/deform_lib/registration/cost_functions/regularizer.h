#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>

#include <vector>

#include "../displacement_field.h"

struct Regularizer
{
    Regularizer(
            const float weight=0.0f,
            const float scale=1.0f,
            float exponent=1.0f,
            const float3& spacing={1.0f, 1.0f, 1.0f})
        : _weight(weight)
        , _scale(scale)
        , _half_exponent(0.5f * exponent)
        , _spacing(spacing)
    {
    }

    virtual ~Regularizer() {}

    void set_regularization_weight(const float weight)
    {
        _weight = weight;
    }
    void set_regularization_scale(const float scale)
    {
        _scale = scale;
    }
    void set_regularization_exponent(const float exponent)
    {
        _half_exponent = 0.5f * exponent;
    }
    void set_fixed_spacing(const float3& spacing)
    {
        _spacing = spacing;
    }

    // Sets the initial displacement for this registration level. This will be
    //  the reference when computing the regularization energy. Any displacement
    //  identical to the initial displacement will result in zero energy.
    void set_initial_displacement(const DisplacementField& initial)
    {
        _initial = initial;
    }

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_weight_map(const stk::VolumeFloat& map) { _weight_map = map; }
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    /// p   : Position in fixed image
    /// def0 : Deformation in active voxel [mm]
    /// def1 : Deformation in neighbor [mm]
    /// step : Direction to neighbor
    inline double operator()(const int3& p, const float3& def0,
                             const float3& def1, const int3& step)
    {
        float3 step_in_mm {
            step.x*_spacing.x,
            step.y*_spacing.y,
            step.z*_spacing.z
        };

        // The diff should be relative to the initial displacement diff
        float3 diff = (def0-_initial.get(p)) - (def1-_initial.get(p+step));

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

    float _weight;
    float _scale;
    float _half_exponent;
    float3 _spacing;

    DisplacementField _initial;

    #ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        stk::VolumeFloat _weight_map;
    #endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP
};

