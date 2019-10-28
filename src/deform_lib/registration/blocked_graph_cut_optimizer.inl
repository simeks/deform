#include "block_change_flags.h"
#include "../config.h"
#include "../profiler/profiler.h"

#include <stk/common/log.h>

#include <functional>
#include <iomanip>

template<
    typename TDisplacementField,
    typename TUnaryTerm,
    typename TBinaryTerm,
    typename TSolver
    >
BlockedGraphCutOptimizer
<TDisplacementField, TUnaryTerm, TBinaryTerm, TSolver>
::BlockedGraphCutOptimizer(
    const std::vector<int3>& neighborhood,
    const int3& block_size,
    double block_energy_epsilon,
    int max_iteration_count) :
    _neighborhood(neighborhood),
    _block_size(block_size),
    _block_energy_epsilon(block_energy_epsilon),
    _max_iteration_count(max_iteration_count)
{
}
template<
    typename TDisplacementField,
    typename TUnaryTerm,
    typename TBinaryTerm,
    typename TSolver
>
BlockedGraphCutOptimizer<TDisplacementField, TUnaryTerm, TBinaryTerm, TSolver>
::~BlockedGraphCutOptimizer()
{
}
template<
    typename TDisplacementField,
    typename TUnaryTerm,
    typename TBinaryTerm,
    typename TSolver
>
void BlockedGraphCutOptimizer<TDisplacementField, TUnaryTerm, TBinaryTerm, TSolver>
::execute(
    TUnaryTerm& unary_fn,
    TBinaryTerm& binary_fn,
    float3 step_size,
    TDisplacementField& df
)
{
    dim3 dims = df.size();

    // Setting the block size to (0, 0, 0) will disable blocking and run the whole volume
    int3 block_dims = _block_size;
    if (block_dims.x == 0) block_dims.x = dims.x;
    if (block_dims.y == 0) block_dims.y = dims.y;
    if (block_dims.z == 0) block_dims.z = dims.z;

    int3 block_count {
        int(dims.x + block_dims.x - 1) / block_dims.x,
        int(dims.y + block_dims.y - 1) / block_dims.y,
        int(dims.z + block_dims.z - 1) / block_dims.z
    };

    DLOG(Info) << "Volume size: " << dims;
    DLOG(Info) << "Block count: " << block_count;
    DLOG(Info) << "Block size: " << block_dims;

    BlockChangeFlags change_flags(block_count);

    int num_iterations = 0;
    LOG(Info) << "Initial Energy: " << calculate_energy(unary_fn, binary_fn, df);

    bool done = false;
    while (!done) {
        // A max_iteration_count of -1 means we run until we converge
        if (_max_iteration_count != -1 && num_iterations >= _max_iteration_count)
            break;

        // TODO: !!!!!
        unary_fn.pre_iteration_hook(num_iterations, df.volume());

        done = true;

        size_t num_blocks_changed = 0;

        for (int use_shift = 0; use_shift < 2; ++use_shift) {
        PROFILER_SCOPE("shift", 0xFF766952);
        if (use_shift == 1 && (block_count.x * block_count.y * block_count.z) <= 1)
            continue;

        /*
            We only do shifting in the directions that requires it
        */

        int3 block_offset{0, 0, 0};
        int3 real_block_count = block_count;
        if (use_shift == 1) {
            block_offset.x = block_count.x == 1 ? 0 : (block_dims.x / 2);
            block_offset.y = block_count.y == 1 ? 0 : (block_dims.y / 2);
            block_offset.z = block_count.z == 1 ? 0 : (block_dims.z / 2);

            if (block_count.x > 1) real_block_count.x += 1;
            if (block_count.y > 1) real_block_count.y += 1;
            if (block_count.z > 1) real_block_count.z += 1;
        }

        for (int black_or_red = 0; black_or_red < 2; black_or_red++) {
            PROFILER_SCOPE("red_black", 0xFF339955);
            int num_blocks = real_block_count.x * real_block_count.y * real_block_count.z;

            #pragma omp parallel for schedule(dynamic) reduction(+:num_blocks_changed)
            for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
                PROFILER_SCOPE("block", 0xFFAA623D);
                int block_x = block_idx % real_block_count.x;
                int block_y = (block_idx / real_block_count.x) % real_block_count.y;
                int block_z = block_idx / (real_block_count.x*real_block_count.y);

                int off = (block_z) % 2;
                off = (block_y + off) % 2;
                off = (block_x + off) % 2;

                if (off != black_or_red) {
                    continue;
                }

                int3 block_p{block_x, block_y, block_z};

                bool need_update = change_flags.is_block_set(block_p, use_shift == 1);
                for (int3 n : _neighborhood) {
                    int3 neighbor = block_p + n;
                    if (0 <= neighbor.x && neighbor.x < real_block_count.x &&
                        0 <= neighbor.y && neighbor.y < real_block_count.y &&
                        0 <= neighbor.z && neighbor.z < real_block_count.z) {
                        need_update = need_update || change_flags.is_block_set(neighbor, use_shift == 1);
                    }
                }

                if (!need_update) {
                    continue;
                }

                bool block_changed = false;
                
                for (int3 n : _neighborhood) {
                    // delta in [mm]
                    float3 delta {
                        step_size.x * n.x,
                        step_size.y * n.y,
                        step_size.z * n.z
                    };

                    block_changed |= do_block(
                        unary_fn,
                        binary_fn,
                        block_p,
                        block_dims,
                        block_offset,
                        delta,
                        df
                    );
                }

                if (block_changed)
                    ++num_blocks_changed;

                change_flags.set_block(block_p, block_changed, use_shift == 1);
            }
        }
        }

        PROFILER_COUNTER_SET("blocks_changed", num_blocks_changed);

        done = num_blocks_changed == 0;

        LOG(Verbose) << "Iteration " << num_iterations << ", "
                     << "Changed " << num_blocks_changed << " blocks, "
                     << "Energy: " << std::fixed << std::setprecision(9)
                                   << calculate_energy(unary_fn, binary_fn, df);

        ++num_iterations;

        PROFILER_FLIP();
    }
    LOG(Info) << "Energy: " << calculate_energy(unary_fn, binary_fn, df) << ", "
              << "Iterations: " << num_iterations;
}

template<
    typename TDisplacementField,
    typename TUnaryTerm,
    typename TBinaryTerm,
    typename TSolver
>
bool BlockedGraphCutOptimizer<TDisplacementField, TUnaryTerm, TBinaryTerm, TSolver>
::do_block(
    TUnaryTerm& unary_fn,
    TBinaryTerm& binary_fn,
    const int3& block_p,
    const int3& block_dims,
    const int3& block_offset,
    const float3& delta, // delta in mm
    TDisplacementField& df
)
{
    dim3 dims = df.size();

    using FlowType = typename TSolver::FlowType;
    TSolver solver(block_dims);

    FlowType current_energy = 0;
    {
    PROFILER_SCOPE("build", 0xFF228844);

    for (int sub_z = 0; sub_z < block_dims.z; ++sub_z) {
    for (int sub_y = 0; sub_y < block_dims.y; ++sub_y) {
    for (int sub_x = 0; sub_x < block_dims.x; ++sub_x) {
        // Global coordinates
        int gx = block_p.x * block_dims.x - block_offset.x + sub_x;
        int gy = block_p.y * block_dims.y - block_offset.y + sub_y;
        int gz = block_p.z * block_dims.z - block_offset.z + sub_z;

        int3 p{gx, gy, gz};
        
        // Skip voxels outside volume
        if (gx < 0 || gx >= int(dims.x) ||
            gy < 0 || gy >= int(dims.y) ||
            gz < 0 || gz >= int(dims.z)) {
            solver.add_term1(sub_x, sub_y, sub_z, 0, 0);
            continue;
        }

        float3 d1 = df.get(p);
        float3 d1d = df.get(p, delta);

        double f0 = unary_fn(p, d1);
        double f1 = unary_fn(p, d1d);

        // Block borders (excl image borders) (T-weights with binary term for neighboring voxels)

        if (sub_x == 0 && gx != 0) {
            int3 step{-1, 0, 0};
            float3 d2 = df.get(p + step);
            f0 += binary_fn(p, d1, d2, step);
            f1 += binary_fn(p, d1d, d2, step);
        }
        else if (sub_x == block_dims.x - 1 && gx < int(dims.x) - 1) {
            int3 step{1, 0, 0};
            float3 d2 = df.get(p + step);
            f0 += binary_fn(p, d1, d2, step);
            f1 += binary_fn(p, d1d, d2, step);
        }

        if (sub_y == 0 && gy != 0) {
            int3 step{0, -1, 0};
            float3 d2 = df.get(p + step);
            f0 += binary_fn(p, d1, d2, step);
            f1 += binary_fn(p, d1d, d2, step);
        }
        else if (sub_y == block_dims.y - 1 && gy < int(dims.y) - 1) {
            int3 step{0, 1, 0};
            float3 d2 = df.get(p + step);
            f0 += binary_fn(p, d1, d2, step);
            f1 += binary_fn(p, d1d, d2, step);
        }

        if (sub_z == 0 && gz != 0) {
            int3 step{0, 0, -1};
            float3 d2 = df.get(p + step);
            f0 += binary_fn(p, d1, d2, step);
            f1 += binary_fn(p, d1d, d2, step);
        }
        else if (sub_z == block_dims.z - 1 && gz < int(dims.z) - 1) {
            int3 step{0, 0, 1};
            float3 d2 = df.get(p + step);
            f0 += binary_fn(p, d1, d2, step);
            f1 += binary_fn(p, d1d, d2, step);
        }

        solver.add_term1(sub_x, sub_y, sub_z, f0, f1);

        current_energy += f0;

        #define ADD_STEP(x_, y_, z_) \
            int3 step = int3{x_, y_, z_}; \
            float3 d2 = df.get(p+step); \
            float3 d2d = df.get(p+step, delta); \
            double f00 = binary_fn(p, d1, d2, step); \
            double f01 = binary_fn(p, d1, d2d, step); \
            double f10 = binary_fn(p, d1d, d2, step); \
            double f11 = binary_fn(p, d1d, d2d, step); \
            solver.add_term2( \
                sub_x, sub_y, sub_z, \
                sub_x + step.x, sub_y + step.y, sub_z + step.z, \
                f00, f01, f10, f11 \
            ); \
            current_energy += f00;

        if (sub_x + 1 < block_dims.x && gx + 1 < int(dims.x)) {
            ADD_STEP(1, 0, 0);
        }
        if (sub_y + 1 < block_dims.y && gy + 1 < int(dims.y)) {
            ADD_STEP(0, 1, 0);
        }
        if (sub_z + 1 < block_dims.z && gz + 1 < int(dims.z)) {
            ADD_STEP(0, 0, 1);
        }
        #undef ADD_STEP
    }
    }
    }
    }

    FlowType current_emin;
    {
        PROFILER_SCOPE("minimize", 0xFF985423);
        current_emin = solver.minimize();
    }

    bool changed_flag = false;

    if (1.0 - current_emin / current_energy > _block_energy_epsilon) // Accept solution
    {
    PROFILER_SCOPE("apply", 0xFF767323);
    for (int sub_z = 0; sub_z < block_dims.z; sub_z++) {
    for (int sub_y = 0; sub_y < block_dims.y; sub_y++) {
    for (int sub_x = 0; sub_x < block_dims.x; sub_x++) {
        // Global coordinates
        int gx = block_p.x * block_dims.x - block_offset.x + sub_x;
        int gy = block_p.y * block_dims.y - block_offset.y + sub_y;
        int gz = block_p.z * block_dims.z - block_offset.z + sub_z;

        // Skip voxels outside volume
        if (gx < 0 || gx >= int(dims.x) ||
            gy < 0 || gy >= int(dims.y) ||
            gz < 0 || gz >= int(dims.z))
        {
            continue;
        }

        if (solver.get_var(sub_x, sub_y, sub_z) == 1) {
            int3 p{gx, gy, gz};
            df.update(p, delta);
            changed_flag = true;
        }
    }
    }
    }
    }


    return changed_flag;
}

template<
    typename TDisplacementField,
    typename TUnaryTerm,
    typename TBinaryTerm,
    typename TSolver
>
double BlockedGraphCutOptimizer<TDisplacementField, TUnaryTerm, TBinaryTerm, TSolver>
::calculate_energy(
    TUnaryTerm& unary_fn,
    TBinaryTerm& binary_fn,
    TDisplacementField& df
)
{
    dim3 dims = df.size();

    double total_energy = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:total_energy)
    for (int gz = 0; gz < int(dims.z); ++gz) {
    for (int gy = 0; gy < int(dims.y); ++gy) {
    for (int gx = 0; gx < int(dims.x); ++gx) {
        int3 p{gx, gy, gz};
        float3 d1 = df.get(p);

        total_energy += unary_fn(p, d1);

        if (gx + 1 < int(dims.x)) {
            int3 step{1, 0, 0};
            float3 d2 = df.get(p + step);
            total_energy += binary_fn(p, d1, d2, step);
        }
        if (gy + 1 < int(dims.y)) {
            int3 step{0, 1, 0};
            float3 d2 = df.get(p + step);
            total_energy += binary_fn(p, d1, d2, step);
        }
        if (gz + 1 < int(dims.z)) {
            int3 step{0, 0, 1};
            float3 d2 = df.get(p + step);
            total_energy += binary_fn(p, d1, d2, step);
        }
    }
    }
    }
    return total_energy;
}

