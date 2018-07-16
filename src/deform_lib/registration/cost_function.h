#pragma once

#include "../config.h"
#include "level_context.h"

#include <stk/common/assert.h>
#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>

#include <memory>
#include <tuple>
#include <vector>

struct Regularizer
{
    Regularizer() :
        _weight(0.0f), _spacing({1.0f, 1.0f, 1.0f})
    {
    }

    // Called at the beginning of each registration level
    //  initial : 
    //  spacing : Fixed volume spacing for this level.
    void begin_level(const LevelContext& ctx)
    {
        // The initial displacement will be used as the reference when computing
        //  the regularization energy. Any displacement identical to the initial
        //  displacement will result in zero energy.
        _initial = ctx.initial_displacement;
        // TODO: Is it safe to assume slot 0 is always set?
        _spacing = ctx.fixed_volumes[0].spacing();
        _weight = ctx.regularization_weight;
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        _weight_map = ctx.regularization_weight_map;
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    }

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
    virtual void begin_level(const LevelContext& ctx) = 0;
    virtual float cost(const int3& p, const float3& def) = 0;
};


struct SoftConstraintsFunction : public SubFunction
{
    SoftConstraintsFunction(float constraint_weight) :
        _constraint_weight(constraint_weight)
    {}

    void begin_level(const LevelContext& ctx)
    {
        _constraint_mask = ctx.constraint_mask;
        _constraint_values = ctx.constraint_values;
        _constraint_values = ctx.constraint_values;
    }

    float cost(const int3& p, const float3& def) 
    {
        if (_constraint_mask(p) != 0)
        {
            float3 diff = def - _constraint_values(p);
            
            // Distance^2 in [mm]
            float dist_squared = stk::norm2(diff);
            
            return std::min(_constraint_weight*dist_squared, 1000.0f); // Clamp to avoid explosion
        }
        return 0.0f;
    }
    stk::VolumeUChar _constraint_mask;
    stk::VolumeFloat3 _constraint_values;
    float _constraint_weight;
};


template<typename T>
struct SquaredDistanceFunction : public SubFunction
{
    // image_slot : Slot for source image data
    SquaredDistanceFunction(int image_slot) : _image_slot(image_slot)
    {}

    void begin_level(const LevelContext& ctx)
    {
        _fixed = ctx.fixed_volumes[_image_slot];
        _moving = ctx.moving_volumes[_image_slot];
    }

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

        float f = fabs(float(_fixed(p) - moving_v));
        return f*f;
    }

    int _image_slot;

    stk::VolumeHelper<T> _fixed;
    stk::VolumeHelper<T> _moving;
};

template<typename T>
struct NCCFunction : public SubFunction
{
    NCCFunction(int image_slot, int radius) : 
        _image_slot(image_slot),
        _radius2(radius*radius)
    {}

    void begin_level(const LevelContext& ctx)
    {
        _fixed = ctx.fixed_volumes[_image_slot];
        _moving = ctx.moving_volumes[_image_slot];
    }

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
                    if (r2 > _radius2)
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

    int _image_slot;
    int _radius2;

    stk::VolumeHelper<T> _fixed;
    stk::VolumeHelper<T> _moving;
};



struct UnaryFunction
{
    struct WeightedFunction {
        float weight;
        std::unique_ptr<SubFunction> function;
    };

    UnaryFunction() : 
        _regularization_weight(0.0f)
    {
    }

    void begin_level(const LevelContext& ctx)
    {
        _regularization_weight = ctx.regularization_weight;
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        _regularization_weight_map = ctx.regularization_weight_map;
#endif
        for (auto& fn : functions) {
            fn.function->begin_level(ctx);
        }
    }

    void add_function(std::unique_ptr<SubFunction> fn, float weight)
    {
        functions.push_back({weight, std::move(fn)});
    }

    inline double operator()(const int3& p, const float3& def)
    {
        double sum = 0.0f;
        for (auto& fn : functions) {
            sum += fn.weight * fn.function->cost(p, def);
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

    std::vector<WeightedFunction> functions;
};
