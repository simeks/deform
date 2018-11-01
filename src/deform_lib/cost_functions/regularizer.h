#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>

#include <vector>

struct Regularizer
{
    Regularizer(float weight=0.0f, const float3& spacing={1.0f, 1.0f, 1.0f}) :
        _weight(weight), _spacing(spacing)
    {
    }

    virtual ~Regularizer() {}

    void set_regularization_weight(float weight)
    {
        _weight = weight;
    }
    void set_fixed_spacing(const float3& spacing)
    {
        _spacing = spacing;
    }

    // Sets the initial displacement for this registration level. This will be
    //  the reference when computing the regularization energy. Any displacement
    //  identical to the initial displacement will result in zero energy.
    void set_initial_displacement(const stk::VolumeFloat3& initial)
    {
        _initial = initial;
    }

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_weight_map(stk::VolumeFloat& map) { _weight_map = map; }
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
        float3 diff = (def0-_initial(p)) - (def1-_initial(p+step));

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

        return w * dist_squared / step_squared;
    }

    float _weight;
    float3 _spacing;

    stk::VolumeFloat3 _initial;

    #ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        stk::VolumeFloat _weight_map;
    #endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP
};

