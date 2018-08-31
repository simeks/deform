#include "gpu_blocked_graph_cut_optimizer.h"


GpuBlockedGraphCutOptimizer::GpuBlockedGraphCutOptimizer(
    const int3& block_size, double block_energy_epsilon) :
        _block_size(block_size),
        _block_energy_epsilon(block_energy_epsilon)
{
}
GpuBlockedGraphCutOptimizer::~GpuBlockedGraphCutOptimizer()
{
}

void GpuBlockedGraphCutOptimizer::execute(
    GpuUnaryFunction& unary_fn,
    GpuBinaryFunction& binary_fn,
    float3 step_size,
    stk::GpuVolume& df
)
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
    //LOG(Info) << "Initial Energy: " << calculate_energy(unary_fn, binary_fn, def);

    // TODO: Merge unary and binary cost into one volume
    stk::VolumeFloat cpu_unary_cost(dims, 0.0f);
    stk::VolumeFloat4 cpu_binary_cost(dims, float4{0});

    GpuVolume unary_cost(cpu_unary_cost);
    GpuVolume binary_cost(cpu_binary_cost);

    stk::VolumeFloat4 cpu_df = df.download();

    bool done = false;
    while (!done) {
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
                
                {
                    PROFILER_SCOPE("gpu_cost");
                    unary_fn(df, unary_cost);
                    binary_fn(df, binary_cost);
                }

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

        PROFILER_COUNTER_SET("blocks_changed", num_blocks_changed);

        done = num_blocks_changed == 0;
        
        ++num_iterations;

        PROFILER_FLIP();
    }
    // LOG(Info) << "Energy: " << calculate_energy(unary_fn, binary_fn, def) 
    //     << ", Iterations: " << num_iterations;
}
