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
    BlockedGraphCutOptimizer(const int3& block_size);
    ~BlockedGraphCutOptimizer();

    /// step_size : Step size in [voxels]
    void execute(
        TUnaryTerm& unary_fn, 
        TBinaryTerm& binary_fn,
        float3 step_size, 
        VolumeFloat3& def);

private:
    float calculate_energy(
        TUnaryTerm& unary_fn,
        TBinaryTerm& binary_fn,
        VolumeFloat3& def
    );

    int3 _block_size;
};

#include "blocked_graph_cut_optimizer.inl"
