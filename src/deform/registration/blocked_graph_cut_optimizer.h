#pragma once

#include <framework/math/float3.h>
#include <framework/math/int3.h>
#include <framework/math/types.h>
#include <framework/volume/volume_helper.h>

const float default_regularization_weight = 0.1f;

struct Regularizer
{
    Regularizer(const float3& fixed_spacing) : _spacing(fixed_spacing) {}

    /// p   : Position in fixed image
    /// def0 : Deformation in active voxel [voxels] (in fixed image space)
    /// def1 : Deformation in neighbor [voxels] (in fixed image space)
    /// step : Direction to neighbor [voxels]
    inline float operator()(const int3& /*p*/, const float3& def0, const float3& def1, const int3& step)
    {
        float3 step_in_mm {step.x*_spacing.x, step.y*_spacing.y, step.z*_spacing.z};
        
        float3 diff = def0 - def1;
        float3 diff_in_mm {diff.x*_spacing.x, step.y*_spacing.y, step.z*_spacing.z};
        
        float dist_squared = math::length_squared(diff_in_mm);
        float step_squared = math::length_squared(step_in_mm);
        return default_regularization_weight * dist_squared / step_squared;
    }

    float3 _spacing;
};

template<typename T>
struct EnergyFunction
{
    EnergyFunction(const VolumeHelper<T>& fixed, const VolumeHelper<T>& moving) : _fixed(fixed), _moving(moving) {} 

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
        return (1 - default_regularization_weight)*powf(fabs(float(_fixed(p) - moving_v)), 2);
    }

    VolumeHelper<T> _fixed;
    VolumeHelper<T> _moving;
};

template<
    typename TUnaryTerm,
    typename TBinaryTerm
    >
class BlockedGraphCutOptimizer
{
public:
    BlockedGraphCutOptimizer();
    ~BlockedGraphCutOptimizer();

    /// step_size : Step size in [voxels]
    void execute(
        TUnaryTerm& unary_fn, 
        TBinaryTerm& binary_fn,
        float3 step_size, 
        VolumeFloat3& def);

private:
    bool do_block(
        TUnaryTerm& unary_fn, 
        TBinaryTerm& binary_fn, 
        const int3& block_p, 
        const int3& block_dims, 
        const int3& block_offset, 
        const float3& delta, // delta in [voxels]
        VolumeFloat3& def);

    int3 _neighbors[6];
    int3 _block_size;
};

#include "blocked_graph_cut_optimizer.inl"
