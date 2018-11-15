#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>
#include <stk/math/types.h>

#include "../config.h"


template<
    typename TUnaryTerm,
    typename TRegularizer,
    typename TSolver
    >
class BlockedGraphCutOptimizer
{
public:
    BlockedGraphCutOptimizer(
        const int3& block_size,
        double block_energy_epsilon,
        int max_iteration_count
    );
    ~BlockedGraphCutOptimizer();

    /// step_size : Step size in [mm]
    void execute(
        TUnaryTerm& unary_fn,
        TRegularizer& regularizer_fn,
        float3 step_size,
        stk::VolumeFloat3& def);

private:
    bool do_block(
        TUnaryTerm& unary_fn,
        TRegularizer& regularizer_fn,
        const int3& block_p,
        const int3& block_dims,
        const int3& block_offset,
        const float3& delta, // delta in [mm]
        stk::VolumeFloat3& def
    );

    double calculate_energy(
        TUnaryTerm& unary_fn,
        TRegularizer& regularizer_fn,
        stk::VolumeFloat3& def
    );

    template<typename FlowType>
    FlowType compute_block_diffusion(
        TUnaryTerm& unary_fn,
        TRegularizer& regularizer_fn,
        TSolver& graph,
        const int3& block_p,
        const int3& block_dims,
        const int3& block_offset,
        const float3& delta,
        stk::VolumeFloat3& def
    );

    template<typename FlowType>
    FlowType compute_block_bending(
        TUnaryTerm& unary_fn,
        TRegularizer& regularizer_fn,
        TSolver& graph,
        const int3& block_p,
        const int3& block_dims,
        const int3& block_offset,
        const float3& delta,
        stk::VolumeFloat3& def
    );

    int3 _neighbors[6];
    int3 _block_size;
    double _block_energy_epsilon;

    // Maximum number of iterations, -1 indicates an infinite number of iterations
    int _max_iteration_count;
};

#include "blocked_graph_cut_optimizer.inl"
