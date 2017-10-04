#pragma once

#include <framework/math/float3.h>
#include <framework/math/int3.h>
#include <framework/volume/volume_helper.h>

struct Regularizer
{
    Regularizer(float weight, const float3& fixed_spacing) : _weight(weight), _spacing(fixed_spacing) {}

    /// p   : Position in fixed image
    /// def0 : Deformation in active voxel [voxels] (in fixed image space)
    /// def1 : Deformation in neighbor [voxels] (in fixed image space)
    /// step : Direction to neighbor [voxels]
    inline float operator()(const int3& /*p*/, const float3& def0, const float3& def1, const int3& step)
    {
        float3 step_in_mm {step.x*_spacing.x, step.y*_spacing.y, step.z*_spacing.z};
        
        float3 diff = def0 - def1;
        float3 diff_in_mm {diff.x*_spacing.x, diff.y*_spacing.y, diff.z*_spacing.z};
        
        float dist_squared = math::length_squared(diff_in_mm);
        float step_squared = math::length_squared(step_in_mm);
        return _weight * dist_squared / step_squared;
    }

    float _weight;
    float3 _spacing;
};

template<typename T>
struct EnergyFunction
{
    EnergyFunction(float weight, const VolumeHelper<T>& fixed, const VolumeHelper<T>& moving) : _weight(weight), _fixed(fixed), _moving(moving) {} 

    /// p   : Position in fixed image
    /// def : Deformation to apply [voxels in fixed image]
    inline float operator()(const int3& p, const float3& def)
    {
        float3 fixed_p{
            float(p.x) + def.x,
            float(p.y) + def.y,
            float(p.z) + def.z
        }; 
        
        // TODO: Cleanup

        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p / _moving.spacing()) - _moving.origin();

        T moving_v = _moving.linear_at(moving_p, volume::Border_Constant);

        // TODO: Float cast
        //printf("(%d, %d, %d) : (%f %f %f)\n", p.x, p.y, p.z, moving_p.x, moving_p.y, moving_p.z);
        return _weight*powf(fabs(float(_fixed(p) - moving_v)), 2);
    }

    float _weight;
    VolumeHelper<T> _fixed;
    VolumeHelper<T> _moving;
};

template<typename T>
struct EnergyFunctionWithConstraints
{
    EnergyFunctionWithConstraints(
        float weight, 
        const VolumeHelper<T>& fixed, 
        const VolumeHelper<T>& moving,
        const VolumeUInt8& constraint_mask,
        const VolumeFloat3& _constraints_values)
        : 
        _weight(weight), 
        _fixed(fixed), 
        _moving(moving),
        _constraints_mask(constraint_mask),
        _constraints_values(_constraints_values) {} 

    /// p   : Position in fixed image
    /// def : Deformation to apply [voxels in fixed image]
    inline float operator()(const int3& p, const float3& def)
    {
        if (_constraints_mask(p) != 0)
        {
            float dist = math::length_squared(def - _constraints_values(p));
            if (dist <= 0.0001f)
                return 0.0f;
            else
                return 10.0f;
        }

        float3 fixed_p{
            float(p.x) + def.x,
            float(p.y) + def.y,
            float(p.z) + def.z
        }; 
        
        // TODO: Cleanup

        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p / _moving.spacing()) - _moving.origin();

        T moving_v = _moving.linear_at(moving_p, volume::Border_Constant);

        // TODO: Float cast
        //printf("(%d, %d, %d) : (%f %f %f)\n", p.x, p.y, p.z, moving_p.x, moving_p.y, moving_p.z);
        return _weight*powf(fabs(float(_fixed(p) - moving_v)), 2);
    }

    float _weight;
    VolumeHelper<T> _fixed;
    VolumeHelper<T> _moving;
    VolumeUInt8 _constraints_mask;
    VolumeFloat3 _constraints_values;
};

