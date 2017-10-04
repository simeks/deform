#include "block_change_flags.h"
#include "config.h"

#include <framework/debug/log.h>
#include <framework/graph_cut/graph_cut.h>
#include <framework/thread/thread.h>


template<
    typename TUnaryTerm,
    typename TBinaryTerm
>
BlockedGraphCutOptimizer<TUnaryTerm, TBinaryTerm>::BlockedGraphCutOptimizer()
{
    _neighbors[0] = {1, 0, 0};
    _neighbors[1] = {-1, 0, 0};
    _neighbors[2] = {0, 1, 0};
    _neighbors[3] = {0, -1, 0};
    _neighbors[4] = {0, 0, 1};
    _neighbors[5] = {0, 0, -1};

    _block_size = {12, 12, 12};
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
    VolumeFloat3& def)
{
    Dims dims = def.size();

    // Setting the block size to (0, 0, 0) will disable blocking and run the whole volume
    int3 block_dims = _block_size;
    if (block_dims.x == 0)
        block_dims.x = dims.width;
    if (block_dims.y == 0)
        block_dims.y = dims.height;
    if (block_dims.z == 0)
        block_dims.z = dims.depth;

    int3 block_count{
        int(dims.width) / block_dims.x,
        int(dims.height) / block_dims.y,
        int(dims.depth) / block_dims.z
    };

    // Rest
    int3 block_rest{
        int(dims.width) % block_dims.x,
        int(dims.height) % block_dims.y,
        int(dims.depth) % block_dims.z
    };

    block_count.x += (block_rest.x > 0 ? 1 : 0);
    block_count.y += (block_rest.y > 0 ? 1 : 0);
    block_count.z += (block_rest.z > 0 ? 1 : 0);

    LOG(Info, "Volume size: %d, %d, %d\n", dims.width, dims.height, dims.depth);
    LOG(Info, "Block count: %d, %d, %d\n", block_count.x, block_count.y, block_count.z);
    LOG(Info, "Block size: %d, %d, %d\n", block_dims.x, block_dims.y, block_dims.z);
    LOG(Info, "Block rest: %d, %d, %d\n", block_rest.x, block_rest.y, block_rest.z);

    BlockChangeFlags change_flags(block_count); 

    bool done = false;
    while (!done)
    {
        done = true;

        for (int use_shift = 0; use_shift < 2; ++use_shift)
        {
            if (use_shift == 1 && (block_count.x * block_count.y * block_count.z) <= 1)
                continue;

            /*
                We only do shifting in the directions that requires it
            */

            int3 block_offset{0, 0, 0};
            if (use_shift == 1)
            {
                block_offset.x = block_count.x == 1 ? 0 : (block_dims.x / 2);
                block_offset.y = block_count.y == 1 ? 0 : (block_dims.y / 2);
                block_offset.z = block_count.z == 1 ? 0 : (block_dims.z / 2);
            }

            int3 real_block_count{
                block_count.x + ((block_count.x > 1 && use_shift == 1) ? 1 : 0),
                block_count.y + ((block_count.y > 1 && use_shift == 1) ? 1 : 0),
                block_count.z + ((block_count.z > 1 && use_shift == 1) ? 1 : 0)
            };

            for (int black_or_red = 0; black_or_red < 2; black_or_red++)
            {
                int num_blocks = real_block_count.x * real_block_count.y * real_block_count.z;

#ifdef DF_DEBUG_BLOCK_CHANGE_COUNT
                volatile long num_blocks_changed = 0;
#endif // DF_DEBUG_BLOCK_CHANGE_COUNT
                #pragma omp parallel for
                for (int block_idx = 0; block_idx < num_blocks; ++block_idx)
                {
                    int block_x = (block_idx % real_block_count.x) % real_block_count.y;
                    int block_y = (block_idx / real_block_count.x) % real_block_count.y;
                    int block_z = block_idx / (real_block_count.x*real_block_count.y);

                    int off = (block_z) % 2;
                    off = (block_y + off) % 2;
                    off = (block_x + off) % 2;

                    if (off != black_or_red)
                    {
                        continue;
                    }

                    int3 block_p{block_x, block_y, block_z};

                    bool need_update = change_flags.is_block_set(block_p, use_shift == 1);
                    int n_count = 6; // Neighbors
                    for (int n = 0; n < n_count; ++n)
                    {
                        need_update = need_update || change_flags.is_block_set(block_p + _neighbors[n], use_shift == 1);
                    }

                    if (!need_update)
                    {
                        continue;
                    }


                    bool block_changed = false;
                    for (int n = 0; n < n_count; ++n)
                    {
                        // delta in [voxels]
                        float3 delta{
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

#ifdef DF_DEBUG_BLOCK_CHANGE_COUNT
                    if (block_changed)
                    {
                        thread::interlocked_increment(&num_blocks_changed);
                    }
#endif // DF_DEBUG_BLOCK_CHANGE_COUNT
                    change_flags.set_block(block_p, block_changed, use_shift == 1);
                    done = done && !block_changed;
                }
#ifdef DF_DEBUG_BLOCK_CHANGE_COUNT
                LOG(Debug, "[num_blocks: %d, use_shift: %d, black_or_red: %d] blocks_changed: %d\n", 
                    num_blocks, use_shift, black_or_red, num_blocks_changed);
#endif // DF_DEBUG_BLOCK_CHANGE_COUNT
            }
        }

#if DF_DEBUG_LEVEL >= 1
        LOG(Debug, "Energy: %.10f\n", calculate_energy(unary_fn, binary_fn, def));
#endif

    }
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
    const float3& delta, // delta in voxels
    VolumeFloat3& def
)
{
    Dims dims = def.size();

    GraphCut<float> graph(block_dims);

    float current_energy = 0;
    {
        for (int sub_z = 0; sub_z < block_dims.z; ++sub_z)
        {
            for (int sub_y = 0; sub_y < block_dims.y; ++sub_y)
            {
                for (int sub_x = 0; sub_x < block_dims.x; ++sub_x)
                {
                    // Global coordinates
                    int gx = block_p.x * block_dims.x - block_offset.x + sub_x;
                    int gy = block_p.y * block_dims.y - block_offset.y + sub_y;
                    int gz = block_p.z * block_dims.z - block_offset.z + sub_z;

                    // Skip voxels outside volume
                    if (gx < 0 || gx >= int(dims.width) ||
                        gy < 0 || gy >= int(dims.height) ||
                        gz < 0 || gz >= int(dims.depth))
                    {
                        graph.add_term1(sub_x, sub_y, sub_z, 0, 0);
                        continue;
                    }

                    int3 p{gx, gy, gz};
                    float3 def1 = def(p);

                    float f0 = unary_fn(p, def1);
                    float f1 = unary_fn(p, def1 + delta);

                    // Block borders (excl image borders) (T-weights with binary term for neighboring voxels)

                    if (sub_x == 0 && gx != 0)
                    {
                        int3 step{-1, 0, 0};
                        float3 def2 = def(gx - 1, gy, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }
                    else if (sub_x == block_dims.x - 1 && gx < int(dims.width) - 1)
                    {
                        int3 step{1, 0, 0};
                        float3 def2 = def(gx + 1, gy, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }

                    if (sub_y == 0 && gy != 0)
                    {
                        int3 step{0, -1, 0};
                        float3 def2 = def(gx, gy - 1, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }
                    else if (sub_y == block_dims.y - 1 && gy < int(dims.height) - 1)
                    {
                        int3 step{0, 1, 0};
                        float3 def2 = def(gx, gy + 1, gz);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }

                    if (sub_z == 0 && gz != 0)
                    {
                        int3 step{0, 0, -1};
                        float3 def2 = def(gx, gy, gz - 1);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }
                    else if (sub_z == block_dims.z - 1 && gz < int(dims.depth) - 1)
                    {
                        int3 step{0, 0, 1};
                        float3 def2 = def(gx, gy, gz + 1);
                        f0 += binary_fn(p, def1, def2, step);
                        f1 += binary_fn(p, def1 + delta, def2, step);
                    }

                    graph.add_term1(sub_x, sub_y, sub_z, f0, f1);

                    current_energy += f0;

                    if (sub_x + 1 < block_dims.x && gx + 1 < int(dims.width))
                    {
                        int3 step{1, 0, 0};
                        float3 def2 = def(p + step);
                        float f_same = binary_fn(p, def1, def2, step);
                        float f01 = binary_fn(p, def1, def2 + delta, step);
                        float f10 = binary_fn(p, def1 + delta, def2, step);

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x + 1, sub_y, sub_z, 
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                    if (sub_y + 1 < block_dims.y && gy + 1 < int(dims.height))
                    {
                        int3 step{0, 1, 0};
                        float3 def2 = def(p + step);
                        float f_same = binary_fn(p, def1, def2, step);
                        float f01 = binary_fn(p, def1, def2 + delta, step);
                        float f10 = binary_fn(p, def1 + delta, def2, step);

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x, sub_y + 1, sub_z,
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                    if (sub_z + 1 < block_dims.z && gz + 1 < int(dims.depth))
                    {
                        int3 step{0, 0, 1};
                        float3 def2 = def(p + step);
                        float f_same = binary_fn(p, def1, def2, step);
                        float f01 = binary_fn(p, def1, def2 + delta, step);
                        float f10 = binary_fn(p, def1 + delta, def2, step);

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

    float current_emin = graph.minimize();
    bool changed_flag = false;

#ifdef DF_DEBUG_VOXEL_CHANGE_COUNT
    int voxels_changed_ = 0;
#endif // DF_DEBUG_VOXEL_CHANGE_COUNT

    if (current_emin /*+ 0.0001f*/ < current_energy) //Accept solution
    {
        for (int sub_z = 0; sub_z < block_dims.z; sub_z++)
        {
            for (int sub_y = 0; sub_y < block_dims.y; sub_y++)
            {
                for (int sub_x = 0; sub_x < block_dims.x; sub_x++)
                {
                    // Global coordinates
                    int gx = block_p.x * block_dims.x - block_offset.x + sub_x;
                    int gy = block_p.y * block_dims.y - block_offset.y + sub_y;
                    int gz = block_p.z * block_dims.z - block_offset.z + sub_z;

                    // Skip voxels outside volume
                    if (gx < 0 || gx >= int(dims.width) ||
                        gy < 0 || gy >= int(dims.height) ||
                        gz < 0 || gz >= int(dims.depth))
                    {
                        continue;
                    }

                    if (graph.get_var(sub_x, sub_y, sub_z) == 1)
                    {
                        def(gx, gy, gz) = def(gx, gy, gz) + delta;
                        changed_flag = true;
#ifdef DF_DEBUG_VOXEL_CHANGE_COUNT
                        ++voxels_changed_;
#endif // DF_DEBUG_VOXEL_CHANGE_COUNT
                    }
                }
            }
        }
    }

#ifdef DF_DEBUG_VOXEL_CHANGE_COUNT
    if (voxels_changed_)
        LOG(Debug, "[voxels changed] delta: %f %f %f, block_p: %d %d %d, n: %d (emin: %f, current_energy: %f)\n", 
            delta.x, delta.y, delta.z, block_p.x, block_p.y, block_p.z, voxels_changed_, current_emin, current_energy);
#endif // DF_DEBUG_VOXEL_CHANGE_COUNT

    return changed_flag;
}


template<
    typename TUnaryTerm,
    typename TBinaryTerm
>
float BlockedGraphCutOptimizer<TUnaryTerm, TBinaryTerm>::calculate_energy(
    TUnaryTerm& unary_fn,
    TBinaryTerm& binary_fn,
    VolumeFloat3& def
)
{
    Dims dims = def.size();

    float total_energy = 0;
    for (int gz = 0; gz < int(dims.depth); ++gz)
    {
        for (int gy = 0; gy < int(dims.height); ++gy)
        {
            for (int gx = 0; gx < int(dims.width); ++gx)
            {
                int3 p{gx, gy, gz};
                float3 def1 = def(p);

                total_energy += unary_fn(p, def1);

                if (gx + 1 < int(dims.width))
                {
                    int3 step{1, 0, 0};
                    float3 def2 = def(p + step);
                    total_energy += binary_fn(p, def1, def2, step);
                }
                if (gy + 1 < int(dims.height))
                {
                    int3 step{0, 1, 0};
                    float3 def2 = def(p + step);
                    total_energy += binary_fn(p, def1, def2, step);
                }
                if (gz + 1 < int(dims.depth))
                {
                    int3 step{0, 0, 1};
                    float3 def2 = def(p + step);
                    total_energy += binary_fn(p, def1, def2, step);
                }
            }
        }
    }
    return total_energy;
}

