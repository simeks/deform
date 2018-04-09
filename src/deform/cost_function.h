#pragma once

#include <framework/debug/sse.h>
#include <framework/math/float3.h>
#include <framework/math/int3.h>
#include <framework/volume/volume_helper.h>

#include <tuple>


struct Regularizer
{
    Regularizer(float weight, const float3& fixed_spacing) : _weight(weight), _spacing(fixex_spacing)
    {
    }

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_weight_map(VolumeFloat& map) { _weight_map = map; }
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    /// p   : Position in fixed image
    /// def0 : Deformation in active voxel [voxels] (in fixed image space)
    /// def1 : Deformation in neighbor [voxels] (in fixed image space)
    /// step : Direction to neighbor [voxels]
    inline float operator()(const int3& , const float3& def0, const float3& def1, const int3& step)
    {
        float3 step_in_mm {step.x*_spacing.x, step.y*_spacing.y, step.z*_spacing.z};
        
        float3 diff = def0 - def1;
        float3 diff_in_mm {diff.x*_spacing.x, diff.y*_spacing.y, diff.z*_spacing.z};
        
        float dist_squared = math::length_squared(diff_in_mm);
        float step_squared = math::length_squared(step_in_mm);
        
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

    VolumeFloat _weight_map;
    float _weight;
    
    float3 _spacing;

};

struct SubFunction
{
    virtual float cost(const int3& p, const float3& def) = 0;
};


struct SoftConstraintsFunction : public SubFunction
{
    SoftConstraintsFunction(const VolumeUInt8& constraint_mask,
                            const VolumeFloat3& constraints_values,
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
            float3 diff_in_mm {diff.x*_spacing.x, diff.y*_spacing.y, diff.z*_spacing.z};
            
            // Distance^2 in [mm]
            float dist_squared = math::length_squared(diff_in_mm);
            
            // y=0.05x^2
            return std::min(_constraints_weight*dist_squared, 1000.0f); // Clamp to avoid explosion
        }
        return 0.0f;
    }
    VolumeUInt8 _constraints_mask;
    VolumeFloat3 _constraints_values;
    float _constraints_weight;
    float3 _spacing;
};


template<typename T>
struct SquaredDistanceFunction : public SubFunction
{
    SquaredDistanceFunction(const VolumeHelper<T>& fixed,
                            const VolumeHelper<T>& moving) :
        _fixed(fixed),
        _moving(moving)
    {}

    float cost(const int3& p, const float3& def)
    {
        float3 fixed_p{
            float(p.x) + def.x,
            float(p.y) + def.y,
            float(p.z) + def.z
        }; 
        
        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p - _moving.origin()) / _moving.spacing();

        T moving_v = _moving.linear_at(moving_p, volume::Border_Constant);

        // [Filip]: Addition for partial-body registrations
        if (moving_p.x<0 || moving_p.x>_moving.size().width || 
            moving_p.y<0 || moving_p.y>_moving.size().height || 
            moving_p.z<0 || moving_p.z>_moving.size().depth) {
            return 0;
        }


        // TODO: Float cast
        float f = fabs(float(_fixed(p) - moving_v));
        return f*f;
    }

    VolumeHelper<T> _fixed;
    VolumeHelper<T> _moving;
};

inline bool is_inside(const Dims& dims, const int3& p)
{
    return (p.x >= 0 && p.x < int(dims.width) && p.y >= 0 && p.y < int(dims.height) && p.z >= 0 && p.z < int(dims.depth));
}

template<typename T>
struct NCCFunction : public SubFunction
{
    NCCFunction(const VolumeHelper<T>& fixed,
                const VolumeHelper<T>& moving) :
        _fixed(fixed),
        _moving(moving)
    {}


    float cost(const int3& p, const float3& def)
    {
        float3 fixed_p{
            float(p.x) + def.x,
            float(p.y) + def.y,
            float(p.z) + def.z
        }; 
        
        // [fixed] -> [world] -> [moving]
        float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
        float3 moving_p = (world_p - _moving.origin()) / _moving.spacing();

        // [Filip]: Addition for partial-body registrations
        if (moving_p.x<0 || moving_p.x>_moving.size().width || 
            moving_p.y<0 || moving_p.y>_moving.size().height || 
            moving_p.z<0 || moving_p.z>_moving.size().depth) {
            return 0;
        }

        T sff = 0.0;
        T smm = 0.0;
        T sfm = 0.0;
        T sf = 0.0;
        T sm = 0.0;
        size_t n = 0;

        for (int dz = -2; dz <= 2; ++dz)
        {
            for (int dy = -2; dy <= 2; ++dy)
            {            
                for (int dx = -2; dx <= 2; ++dx)
                {
                    int3 fp{p.x + dx, p.y + dy, p.z + dz};
                    
                    if (!is_inside(_fixed.size(), fp))
                        continue;

                    float3 mp{moving_p.x + dx, moving_p.y + dy, moving_p.z + dz};

                    T fixed_v = _fixed.at(fp, volume::Border_Constant);
                    T moving_v = _moving.linear_at(mp, volume::Border_Constant);

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
            return 1.0;

        // Subtract mean
        sff -= (sf * sf / n);
        smm -= (sm * sm / n);
        sfm -= (sf * sm / n);
        
        double d = sqrt(sff*smm);

        if(d > 1e-14 )
        {
            return 1.0f - float(sfm / d);
        }
        return 1.0;
    }

    VolumeHelper<T> _fixed;
    VolumeHelper<T> _moving;
    VolumeHelper<T> _transformed;
};



struct UnaryFunction
{
    UnaryFunction(float regularization_weight) : 
        _regularization_weight(regularization_weight)
    {
        for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT+1; ++i)
            functions[i] = NULL;
        num_functions = 0;
    }
    void set_regularization_weight_map(VolumeFloat& map) 
    {
        _regularization_weight_map = map;
    }

    void add_function(SubFunction* fn)
    {
        assert(num_functions < DF_MAX_IMAGE_PAIR_COUNT);
        functions[num_functions++] = fn;
    }

    inline float operator()(const int3& p, const float3& def)
    {
        float sum = 0.0f;
        for (int i = 0; i < num_functions; ++i)
        {
            if (functions[i] == NULL) continue;

            sum += functions[i]->cost(p, def);
        }

        float w = _regularization_weight;
        if (_regularization_weight_map.valid())
            w = _regularization_weight_map(p);

        return (1.0f-w)*sum;
    }

    float _regularization_weight;
    VolumeFloat _regularization_weight_map;

    SubFunction* functions[DF_MAX_IMAGE_PAIR_COUNT+1]; // +1 for constraints function
    int num_functions;
};


