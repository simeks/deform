#pragma once

#include "config.h"

#include <stk/common/assert.h>
#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>

#include <memory>
#include <tuple>
#include <vector>

struct Regularizer
{
    Regularizer(float weight=0.0f, const float3& spacing={1.0f, 1.0f, 1.0f}) :
        _weight(weight), _spacing(spacing)
    {
    }

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

struct SubFunction
{
    virtual float cost(const int3& p, const float3& def) = 0;
};


struct SoftConstraintsFunction : public SubFunction
{
    SoftConstraintsFunction(const stk::VolumeUChar& constraint_mask,
                            const stk::VolumeFloat3& constraints_values,
                            float constraints_weight) :
        _constraints_mask(constraint_mask),
        _constraints_values(constraints_values),
        _constraints_weight(constraints_weight),
        _spacing(_constraints_values.spacing())
    {}

    float cost(const int3& p, const float3& def) 
    {
        if (_constraints_mask(p) != 0)
        {
            float3 diff = def - _constraints_values(p);
            
            // Distance^2 in [mm]
            float dist_squared = stk::norm2(diff);
            
            return std::min(_constraints_weight*dist_squared, 1000.0f); // Clamp to avoid explosion
        }
        return 0.0f;
    }
    stk::VolumeUChar _constraints_mask;
    stk::VolumeFloat3 _constraints_values;
    float _constraints_weight;
    float3 _spacing;
};


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
        float3 fixed_p{
            float(p.x),
            float(p.y),
            float(p.z)
        };
        
        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p + def - _moving.origin()) / _moving.spacing();

        T moving_v = _moving.linear_at(moving_p, stk::Border_Constant);

        // [Filip]: Addition for partial-body registrations
        if (moving_p.x<0 || moving_p.x>_moving.size().x || 
            moving_p.y<0 || moving_p.y>_moving.size().y || 
            moving_p.z<0 || moving_p.z>_moving.size().z) {
            return 0;
        }


        // TODO: Float cast
        float f = fabs(float(_fixed(p) - moving_v));
        return f*f;
    }

    stk::VolumeHelper<T> _fixed;
    stk::VolumeHelper<T> _moving;
};

template<typename T>
struct NCCFunction : public SubFunction
{
    NCCFunction(const stk::VolumeHelper<T>& fixed,
                const stk::VolumeHelper<T>& moving) :
        _fixed(fixed),
        _moving(moving)
    {}


    float cost(const int3& p, const float3& def)
    {
        float3 fixed_p{
            float(p.x),
            float(p.y),
            float(p.z)
        }; 
        
        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p + def - _moving.origin()) / _moving.spacing();

        // [Filip]: Addition for partial-body registrations
        if (moving_p.x<0 || moving_p.x>_moving.size().x || 
            moving_p.y<0 || moving_p.y>_moving.size().y || 
            moving_p.z<0 || moving_p.z>_moving.size().z) {
            return 0;
        }

        double sff = 0.0;
        double smm = 0.0;
        double sfm = 0.0;
        double sf = 0.0;
        double sm = 0.0;
        size_t n = 0;

        for (int dz = -2; dz <= 2; ++dz) {
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    // TODO: Does not account for anisotropic volumes
                    int r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 > 4)
                        continue;

                    int3 fp{p.x + dx, p.y + dy, p.z + dz};
                    
                    if (!stk::is_inside(_fixed.size(), fp))
                        continue;

                    float3 mp{moving_p.x + dx, moving_p.y + dy, moving_p.z + dz};

                    T fixed_v = _fixed(fp);
                    T moving_v = _moving.linear_at(mp, stk::Border_Constant);

                    sff += fixed_v * fixed_v;
                    smm += moving_v * moving_v;
                    sfm += fixed_v*moving_v;
                    sm += moving_v;
                    sf += fixed_v;

                    ++n;
                }
            }
        }

        if (n == 0)
            return 0.0f;

        // Subtract mean
        sff -= (sf * sf / n);
        smm -= (sm * sm / n);
        sfm -= (sf * sm / n);
        
        double d = sqrt(sff*smm);

        if(d > 1e-14) {
            return 0.5f*(1.0f-float(sfm / d));
        }
        return 0.0f;
    }

    stk::VolumeHelper<T> _fixed;
    stk::VolumeHelper<T> _moving;
};



struct UnaryFunction
{
    UnaryFunction(float regularization_weight=0.0f) : 
        _regularization_weight(regularization_weight)
    {
    }
    void set_regularization_weight(float weight)
    {
        _regularization_weight = weight;
    }
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_regularization_weight_map(stk::VolumeFloat& map) 
    {
        _regularization_weight_map = map;
    }
#endif

    void add_function(std::unique_ptr<SubFunction> fn)
    {
        functions.push_back(std::move(fn));
    }

    inline double operator()(const int3& p, const float3& def)
    {
        double sum = 0.0f;
        for (auto& fn : functions) {
            sum += fn->cost(p, def);
        }

        float w = _regularization_weight;
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        if (_regularization_weight_map.valid())
            w = _regularization_weight_map(p);
#endif

        return (1.0f-w)*sum;
    }

    float _regularization_weight;
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    stk::VolumeFloat _regularization_weight_map;
#endif

    std::vector<std::unique_ptr<SubFunction>> functions;
};


