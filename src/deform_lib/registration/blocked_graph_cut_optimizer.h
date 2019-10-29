#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>
#include <stk/math/types.h>

#include "../config.h"
#include "displacement_field.h"
#include "settings.h"

#include <vector>

template<
    typename TUnaryTerm,
    typename TBinaryTerm,
    typename TSolver
>
class BlockedGraphCutOptimizer
{
public:
    BlockedGraphCutOptimizer(
        const std::vector<int3>& neighborhood,
        const int3& block_size,
        double block_energy_epsilon,
        int max_iteration_count
    );
    ~BlockedGraphCutOptimizer();

    /// step_size : Step size in [mm]
    void execute(
        TUnaryTerm& unary_fn,
        TBinaryTerm& binary_fn,
        float3 step_size,
        DisplacementField& df
    );

private:
    bool do_block(
        TUnaryTerm& unary_fn,
        TBinaryTerm& binary_fn,
        const int3& block_p,
        const int3& block_dims,
        const int3& block_offset,
        const float3& delta, // delta in [mm]
        DisplacementField& df
    );

    double calculate_energy(
        TUnaryTerm& unary_fn,
        TBinaryTerm& binary_fn,
        DisplacementField& df
    );

    std::vector<int3> _neighborhood;
    int3 _block_size;
    double _block_energy_epsilon;

    // Maximum number of iterations, -1 indicates an infinite number of iterations
    int _max_iteration_count;
};

#include "blocked_graph_cut_optimizer.inl"
