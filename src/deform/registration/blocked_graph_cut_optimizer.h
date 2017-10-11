#pragma once

#include <framework/math/float3.h>
#include <framework/math/int3.h>
#include <framework/math/types.h>
#include <framework/volume/volume_helper.h>

#include "config.h"

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
        VolumeFloat3& def
#ifdef DF_ENABLE_BLOCK_ENERGY_CHECK
        , float& out_block_energy
#endif
    );

    float calculate_energy(
        TUnaryTerm& unary_fn,
        TBinaryTerm& binary_fn,
        VolumeFloat3& def
    );

    int3 _neighbors[6];
    int3 _block_size;
};

#include "blocked_graph_cut_optimizer.inl"
