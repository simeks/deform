#pragma once

#include <framework/math/float3.h>
#include <framework/math/int3.h>
#include <framework/volume/volume_helper.h>

#include <tuple>

/*
    Tissue-specific regularization
    Per edge weight: Mean term of neighboring voxels 

    w = 0.5f*(weights(p) + weights(p+step)) 
*/

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

struct FatWaterRegularizer
{
    FatWaterRegularizer(const VolumeHelper<float>& fixed_water) : 
        _spacing(fixed_water.spacing()), _fixed_water(fixed_water) {}

    /// p   : Position in fixed image
    /// def0 : Deformation in active voxel [voxels] (in fixed image space)
    /// def1 : Deformation in neighbor [voxels] (in fixed image space)
    /// step : Direction to neighbor [voxels]
    inline float operator()(const int3& p, const float3& def0, const float3& def1, const int3& step)
    {
        float3 step_in_mm {step.x*_spacing.x, step.y*_spacing.y, step.z*_spacing.z};
        
        float3 diff = def0 - def1;
        float3 diff_in_mm {diff.x*_spacing.x, diff.y*_spacing.y, diff.z*_spacing.z};
        
        float dist_squared = math::length_squared(diff_in_mm);
        float step_squared = math::length_squared(step_in_mm);

        // Fat = 0.08
        // Water = 0.1

        // w = 0.08 + 0.02 + avg(water, water+step)

        float water1 = fixed_water(p); // Water should be 0-1
        float water2 = fixed_water(p+step);

        float w = 0.08f + 0.02f * 0.5f * (water1+water2)
        return w * dist_squared / step_squared;
    }

    float3 _spacing;
    VolumeHelper<float> _fixed_water;
};

template<typename T>
struct SquaredDistanceFunction
{
    SquaredDistanceFunction(const VolumeHelper<T>& fixed,
                            const VolumeHelper<T>& moving) :
        _fixed(fixed),
        _moving(moving)
    {}

    inline float operator()(const int3& p, const float3& def)
    {
        float3 fixed_p{
            float(p.x) + def.x,
            float(p.y) + def.y,
            float(p.z) + def.z
        }; 
        
        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p / _moving.spacing()) - _moving.origin();

        T moving_v = _moving.linear_at(moving_p, volume::Border_Constant);

        // TODO: Float cast
        return powf(fabs(float(_fixed(p) - moving_v)), 2);
    }

    VolumeHelper<T> _fixed;
    VolumeHelper<T> _moving;
};

struct ConstraintsFunction
{
    ConstraintsFunction(const VolumeUInt8& constraint_mask,
                        const VolumeFloat3& constraints_values) :
        _constraints_mask(constraint_mask),
        _constraints_values(constraints_values)
    {}

    inline float operator()(const int3& p, const float3& def)
    {
        if (_constraints_mask(p) != 0)
        {
            // Distance in voxels
            float dist = math::length_squared(def - _constraints_values(p));
            if (dist <= 0.0001f)
                return 0.0f;
            else
                return 1000.0f;
        }
        return 0.0f;
    }
    VolumeUInt8 _constraints_mask;
    VolumeFloat3 _constraints_values;
};

struct SoftConstraintsFunction
{
    SoftConstraintsFunction(const VolumeUInt8& constraint_mask,
                            const VolumeFloat3& constraints_values) :
        _constraints_mask(constraint_mask),
        _constraints_values(constraints_values),
        _spacing(_constraints_values.spacing())
    {}

    inline float operator()(const int3& p, const float3& def)
    {
        if (_constraints_mask(p) != 0)
        {
            float3 diff = def - _constraints_values(p);
            float3 diff_in_mm {diff.x*_spacing.x, diff.y*_spacing.y, diff.z*_spacing.z};
            
            // Distance^2 in [mm]
            float dist_squared = math::length_squared(diff_in_mm);
            
            // y=0.05x^2
            return std::min(0.05f*dist_squared, 1000.0f); // Clamp to avoid explosion
        }
        return 0.0f;
    }
    VolumeUInt8 _constraints_mask;
    VolumeFloat3 _constraints_values;
    float3 _spacing;
};



template<typename T, size_t I>
struct evaluate_functions
{
    static float eval(T& func, const int3& p, const float3& def)
    {
        return std::get<I>(func)(p, def) + 
            evaluate_functions<T, I-1>::eval(func, p, def);
    }
};

template<typename T>
struct evaluate_functions<T, 0>
{
    static float eval(T& func, const int3& p, const float3& def)
    {
        return std::get<0>(func)(p, def);
    }
};

template<typename ... TFunction>
struct UnaryFunction
{
    UnaryFunction(
        float weight,
        TFunction... functions
    ) : 
        _weight(weight),
        _functions(functions...)
    {}
    
    /// p   : Position in fixed image
    /// def : Deformation to apply [voxels in fixed image]
    inline float operator()(const int3& p, const float3& def)
    {
        return _weight * 
            evaluate_functions<std::tuple<TFunction...>, sizeof...(TFunction)-1>::eval(
                _functions, p, def
        );
    }

    float _weight;
    std::tuple<TFunction...> _functions;
};




