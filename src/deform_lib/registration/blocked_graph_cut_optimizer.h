#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>
#include <stk/math/types.h>

#include "../config.h"


template<
    typename TUnaryTerm,
    typename TBinaryTerm
    >
class BlockedGraphCutOptimizer
{
public:
    BlockedGraphCutOptimizer(const int3& block_size, double block_energy_epsilon);
    ~BlockedGraphCutOptimizer();

    /// step_size : Step size in [mm]
    void execute(
        TUnaryTerm& unary_fn, 
        TBinaryTerm& binary_fn,
        float step_size, 
        stk::VolumeFloat3& def);

private:
    bool do_block(
        TUnaryTerm& unary_fn,
        TBinaryTerm& binary_fn,
        const int3& block_p, 
        const int3& block_dims, 
        const int3& block_offset, 
        const float3& delta, // delta in [mm]
        stk::VolumeFloat3& def
    );

    double calculate_energy(
        TUnaryTerm& unary_fn,
        TBinaryTerm& binary_fn,
        stk::VolumeFloat3& def
    );

    int3 _neighbors[6];
    int3 _block_size;
    double _block_energy_epsilon;
};

#include "blocked_graph_cut_optimizer.inl"
