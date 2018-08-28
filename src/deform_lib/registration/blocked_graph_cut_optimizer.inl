#include "block_change_flags.h"
#include "../config.h"
#include "../graph_cut/graph_cut.h"

#include <stk/common/log.h>

template<
    typename TUnaryTerm,
    typename TBinaryTerm
>
BlockedGraphCutOptimizer<TUnaryTerm, TBinaryTerm>::BlockedGraphCutOptimizer(
    const int3& block_size,
    double block_energy_epsilon,
    int max_iteration_count) :
    _block_size(block_size),
    _block_energy_epsilon(block_energy_epsilon),
    _max_iteration_count(max_iteration_count)
{
    _neighbors[0] = {1, 0, 0};
    _neighbors[1] = {-1, 0, 0};
    _neighbors[2] = {0, 1, 0};
    _neighbors[3] = {0, -1, 0};
    _neighbors[4] = {0, 0, 1};
    _neighbors[5] = {0, 0, -1};
}
template<
    typename TUnaryTerm,
    typename TBinaryTerm
    >
BlockedGraphCutOptimizer<TUnaryTerm, TBinaryTerm>::~BlockedGraphCutOptimizer()
{
}
template<
    typename TUnaryTerm,
    typename TBinaryTerm
    >
void BlockedGraphCutOptimizer<TUnaryTerm, TBinaryTerm>::execute(
    TUnaryTerm& unary_fn, 
    TBinaryTerm& binary_fn,
    float3 step_size,
    stk::VolumeFloat3& def)
{
    dim3 dims = def.size();

    // Setting the block size to (0, 0, 0) will disable blocking and run the whole volume
    int3 block_dims = _block_size;
    if (block_dims.x == 0)
        block_dims.x = dims.x;
    if (block_dims.y == 0)
        block_dims.y = dims.y;
    if (block_dims.z == 0)
        block_dims.z = dims.z;

    int3 block_count {
        int(dims.x) / block_dims.x,
        int(dims.y) / block_dims.y,
        int(dims.z) / block_dims.z
    };

    // Rest
    int3 block_rest {
        int(dims.x) % block_dims.x,
        int(dims.y) % block_dims.y,
        int(dims.z) % block_dims.z
    };

    block_count.x += (block_rest.x > 0 ? 1 : 0);
    block_count.y += (block_rest.y > 0 ? 1 : 0);
    block_count.z += (block_rest.z > 0 ? 1 : 0);

    DLOG(Info) << "Volume size: " << dims;
    DLOG(Info) << "Block count: " << block_count;
    DLOG(Info) << "Block size: " << block_dims;
    DLOG(Info) << "Block rest: " << block_rest;

    BlockChangeFlags change_flags(block_count); 

    int num_iterations = 0;
    LOG(Info) << "Initial Energy: " << calculate_energy(unary_fn, binary_fn, def);

    bool done = false;
    while (!done) {

        unary_fn.pre_iteration_hook(num_iterations, def);

        done = true;

        size_t num_blocks_changed = 0;

        for (int use_shift = 0; use_shift < 2; ++use_shift) {
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
                int num_blocks = real_block_count.x * real_block_count.y * real_block_count.z;
                
                #pragma omp parallel for schedule(dynamic) reduction(+:num_blocks_changed)
                for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
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
                    int n_count = 6; // Neighbors
                    for (int n = 0; n < n_count; ++n) {
                        int3 neighbor = block_p + _neighbors[n];
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
                    for (int n = 0; n < n_count; ++n) {
                        // delta in [mm]
                        float3 delta {
                            step_size.x * _neighbors[n].x,
                            step_size.y * _neighbors[n].y,
                            step_size.z * _neighbors[n].z
                        };

                        block_changed |= do_block(
                            unary_fn,
                            binary_fn,
                            block_p,
                            block_dims,
                            block_offset,
                            delta,
                            def
                        );
                    }

                    if (block_changed)
                        ++num_blocks_changed;

                    change_flags.set_block(block_p, block_changed, use_shift == 1);
                }
            }
        }

        done = num_blocks_changed == 0;
        
        #ifdef DF_OUTPUT_VOLUME_ENERGY
            LOG(Info) << "Energy: " << calculate_energy(unary_fn, binary_fn, def);
        #endif

        ++num_iterations;

        // A max_iteration_count of -1 means we run until we converge
        if (_max_iteration_count != -1 && num_iterations >= _max_iteration_count)
            break;
    }
    LOG(Info) << "Energy: " << calculate_energy(unary_fn, binary_fn, def) 
        << ", Iterations: " << num_iterations;
}

template<
    typename TUnaryTerm,
    typename TBinaryTerm
>
bool BlockedGraphCutOptimizer<TUnaryTerm, TBinaryTerm>::do_block(
    TUnaryTerm& unary_fn,
    TBinaryTerm& binary_fn,
    const int3& block_p,
    const int3& block_dims,
    const int3& block_offset,
    const float3& delta, // delta in mm
    stk::VolumeFloat3& def
)
{
    dim3 dims = def.size();

    typedef double FlowType;

    GraphCut<FlowType> graph(block_dims);

    FlowType current_energy = 0;
    {
        for (int sub_z = 0; sub_z < block_dims.z; ++sub_z) {
            for (int sub_y = 0; sub_y < block_dims.y; ++sub_y) {
                for (int sub_x = 0; sub_x < block_dims.x; ++sub_x) {
                    // Global coordinates
                    int gx = block_p.x * block_dims.x - block_offset.x + sub_x;
                    int gy = block_p.y * block_dims.y - block_offset.y + sub_y;
                    int gz = block_p.z * block_dims.z - block_offset.z + sub_z;

                    // Skip voxels outside volume
                    if (gx < 0 || gx >= int(dims.x) ||
                        gy < 0 || gy >= int(dims.y) ||
                        gz < 0 || gz >= int(dims.z)) {
                        graph.add_term1(sub_x, sub_y, sub_z, 0, 0);
                        continue;
                    }

                    int3 p{gx, gy, gz};
                    float3 def1 = def(p);
            
                    double f0 = unary_fn(p, def1);
                    double f1 = unary_fn(p, def1 + delta);

                    // Block borders (excl image borders) (T-weights with binary term for neighboring voxels)

                    if (sub_x == 0 && gx != 0) {
                        int3 step{-1, 0, 0};
                        float3 def2 = def(gx - 1, gy, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }
                    else if (sub_x == block_dims.x - 1 && gx < int(dims.x) - 1) {
                        int3 step{1, 0, 0};
                        float3 def2 = def(gx + 1, gy, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }

                    if (sub_y == 0 && gy != 0) {
                        int3 step{0, -1, 0};
                        float3 def2 = def(gx, gy - 1, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }
                    else if (sub_y == block_dims.y - 1 && gy < int(dims.y) - 1) {
                        int3 step{0, 1, 0};
                        float3 def2 = def(gx, gy + 1, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }

                    if (sub_z == 0 && gz != 0) {
                        int3 step{0, 0, -1};
                        float3 def2 = def(gx, gy, gz - 1);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }
                    else if (sub_z == block_dims.z - 1 && gz < int(dims.z) - 1) {
                        int3 step{0, 0, 1};
                        float3 def2 = def(gx, gy, gz + 1);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }

                    graph.add_term1(sub_x, sub_y, sub_z, f0, f1);

                    current_energy += f0;

                    if (sub_x + 1 < block_dims.x && gx + 1 < int(dims.x)) {
                        int3 step{1, 0, 0};
                        float3 def2 = def(p + step);
                        double f_same = binary_fn(p, def1, def2, step);
                        double f01 = binary_fn(p, def1, def2 + delta, step);
                        double f10 = binary_fn(p, def1 + delta, def2, step);

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x + 1, sub_y, sub_z, 
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                    if (sub_y + 1 < block_dims.y && gy + 1 < int(dims.y)) {
                        int3 step{0, 1, 0};
                        float3 def2 = def(p + step);
                        double f_same = binary_fn(p, def1, def2, step);
                        double f01 = binary_fn(p, def1, def2 + delta, step);
                        double f10 = binary_fn(p, def1 + delta, def2, step);

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x, sub_y + 1, sub_z,
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                    if (sub_z + 1 < block_dims.z && gz + 1 < int(dims.z)) {
                        int3 step{0, 0, 1};
                        float3 def2 = def(p + step);
                        double f_same = binary_fn(p, def1, def2, step);
                        double f01 = binary_fn(p, def1, def2 + delta, step);
                        double f10 = binary_fn(p, def1 + delta, def2, step);

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x, sub_y, sub_z + 1,
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                }
            }
        }
    }


    FlowType current_emin;
    {
        current_emin = graph.minimize();
    }

    bool changed_flag = false;

    if (1.0 - current_emin / current_energy > _block_energy_epsilon) // Accept solution
    {
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

                    if (graph.get_var(sub_x, sub_y, sub_z) == 1)
                    {
                        def(gx, gy, gz) = def(gx, gy, gz) + delta;
                        changed_flag = true;
                    }
                }
            }
        }
    }


    return changed_flag;
}

template<
    typename TUnaryTerm,
    typename TBinaryTerm
>
double BlockedGraphCutOptimizer<TUnaryTerm, TBinaryTerm>::calculate_energy(
    TUnaryTerm& unary_fn,
    TBinaryTerm& binary_fn,
    stk::VolumeFloat3& def
)
{
    dim3 dims = def.size();

    double total_energy = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:total_energy)
    for (int gz = 0; gz < int(dims.z); ++gz) {
        for (int gy = 0; gy < int(dims.y); ++gy) {
            for (int gx = 0; gx < int(dims.x); ++gx) {
                int3 p{gx, gy, gz};
                float3 def1 = def(p);

                total_energy += unary_fn(p, def1);

                if (gx + 1 < int(dims.x)) {
                    int3 step{1, 0, 0};
                    float3 def2 = def(p + step);
                    total_energy += binary_fn(p, def1, def2, step);
                }
                if (gy + 1 < int(dims.y)) {
                    int3 step{0, 1, 0};
                    float3 def2 = def(p + step);
                    total_energy += binary_fn(p, def1, def2, step);
                }
                if (gz + 1 < int(dims.z)) {
                    int3 step{0, 0, 1};
                    float3 def2 = def(p + step);
                    total_energy += binary_fn(p, def1, def2, step);
                }
            }
        }
    }
    return total_energy;
}

