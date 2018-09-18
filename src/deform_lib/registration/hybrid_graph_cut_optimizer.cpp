#include "block_change_flags.h"
#include "gpu/cost_function.h"
#include "hybrid_graph_cut_optimizer.h"

#include "deform_lib/graph_cut/graph_cut.h"
#include "deform_lib/profiler/profiler.h"

#include <stk/cuda/stream.h>
#include <stk/image/gpu_volume.h>
#include <stk/io/io.h>

HybridGraphCutOptimizer::HybridGraphCutOptimizer()
{
    _neighbors[0] = {1, 0, 0};
    _neighbors[1] = {-1, 0, 0};
    _neighbors[2] = {0, 1, 0};
    _neighbors[3] = {0, -1, 0};
    _neighbors[4] = {0, 0, 1};
    _neighbors[5] = {0, 0, -1};
}
HybridGraphCutOptimizer::~HybridGraphCutOptimizer()
{
}

void HybridGraphCutOptimizer::execute(
    const Settings::Level& settings,
    GpuUnaryFunction& unary_fn,
    GpuBinaryFunction& binary_fn,
    stk::GpuVolume& df
)
{
    dim3 dims = df.size();

    allocate_cost_buffers(dims);

    // Setting the block size to (0, 0, 0) will disable blocking and run the whole volume
    int3 block_dims = settings.block_size;
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
    //LOG(Info) << "Initial Energy: " << calculate_energy(unary_fn, binary_fn, def);

    bool done = false;
    while (!done) {
        PROFILER_SCOPE("iteration", 0xFF39842A);

        // A max_iteration_count of -1 means we run until we converge
        if (settings.max_iteration_count != -1 && num_iterations >= settings.max_iteration_count)
            break;
        
        done = true;
        size_t num_blocks_changed = 0;

        for (int use_shift = 0; use_shift < 2; ++use_shift) {
            PROFILER_SCOPE("shift", 0xFF766952);
            if (use_shift == 1 && (block_count.x * block_count.y * block_count.z) <= 1)
                continue;

            // We only do shifting in the directions that requires it
            int3 block_offset{0, 0, 0};
            int3 real_block_count = block_count;
            if (use_shift == 1) {
                block_offset.x = block_count.x == 1 ? 0 : (block_dims.x / 2);
                block_offset.y = block_count.y == 1 ? 0 : (block_dims.y / 2);
                block_offset.z = block_count.z == 1 ? 0 : (block_dims.z / 2);
                
                // Only add an additional shift block if necessary, some configurations may not
                //  need the additional block. E.g. a volume of size 5 with a block size of 4 will
                //  cover whole volume with 2 blocks both with and without shifting.

                if (block_count.x > 1 && (block_count.x * block_dims.x - block_offset.x) <= (int)dims.x) 
                    real_block_count.x += 1;
                if (block_count.y > 1 && (block_count.y * block_dims.y - block_offset.y) <= (int)dims.y)
                    real_block_count.y += 1;
                if (block_count.z > 1 && (block_count.z * block_dims.z - block_offset.z) <= (int)dims.z)
                    real_block_count.z += 1;
            }
            
            for (int black_or_red = 0; black_or_red < 2; black_or_red++) {
                PROFILER_SCOPE("red_black", 0xFF339955);
                int num_blocks = real_block_count.x * real_block_count.y * real_block_count.z;
                
                const int n_count = 6; // Neighbors
                for (int n = 0; n < n_count; ++n) {
                    PROFILER_SCOPE("step", 0xFFAA6FE2);

                    // delta in [mm]
                    float3 delta {
                        settings.step_size.x * _neighbors[n].x,
                        settings.step_size.y * _neighbors[n].y,
                        settings.step_size.z * _neighbors[n].z
                    };
                    
                    // Queue all blocks
                    for (int b = 0; b < num_blocks; ++b) {
                        int block_x = b % real_block_count.x;
                        int block_y = (b / real_block_count.x) % real_block_count.y;
                        int block_z = b / (real_block_count.x*real_block_count.y);

                        int off = (block_z) % 2;
                        off = (block_y + off) % 2;
                        off = (block_x + off) % 2;

                        if (off != black_or_red) {
                            continue;
                        }

                        int3 block_idx{block_x, block_y, block_z};

                        bool need_update = change_flags.is_block_set(block_idx, use_shift == 1);
                        for (int i = 0; i < n_count; ++i) {
                            int3 neighbor = block_idx + _neighbors[i];
                            if (0 <= neighbor.x && neighbor.x < real_block_count.x &&
                                0 <= neighbor.y && neighbor.y < real_block_count.y &&
                                0 <= neighbor.z && neighbor.z < real_block_count.z) {
                                need_update = need_update || change_flags.is_block_set(neighbor, use_shift == 1);
                            }
                        }

                        if (!need_update) {
                            continue;
                        }

                        int3 offset{
                            block_x * block_dims.x - block_offset.x,
                            block_y * block_dims.y - block_offset.y,
                            block_z * block_dims.z - block_offset.z
                        };

                        Block block;
                        block.begin = int3{
                            std::max<int>(0, offset.x),
                            std::max<int>(0, offset.y),
                            std::max<int>(0, offset.z)
                        };
                        block.end = int3{
                            std::min<int>(offset.x + block_dims.x, dims.x),
                            std::min<int>(offset.y + block_dims.y, dims.y),
                            std::min<int>(offset.z + block_dims.z, dims.z)
                        };

                        enqueue_block(block);
                    }
                    num_blocks_changed += dispatch_blocks(
                        unary_fn, binary_fn, delta, settings.block_energy_epsilon, df
                    );
                }
            }
        }

        done = num_blocks_changed == 0;

        ++num_iterations;

        PROFILER_FLIP();
    }
    LOG(Info) << "Energy: " << /*calculate_energy(unary_fn, binary_fn, def)*/ "TODO" 
        << ", Iterations: " << num_iterations;
}
void HybridGraphCutOptimizer::allocate_cost_buffers(const dim3& size)
{
    _unary_cost = stk::VolumeFloat2(size, {0});
    _binary_cost_x = stk::VolumeFloat4(size, {0});
    _binary_cost_y = stk::VolumeFloat4(size, {0});
    _binary_cost_z = stk::VolumeFloat4(size, {0});

    _gpu_unary_cost = stk::GpuVolume(size, stk::Type_Float2);
    _gpu_binary_cost_x = stk::GpuVolume(size, stk::Type_Float4);
    _gpu_binary_cost_y = stk::GpuVolume(size, stk::Type_Float4);
    _gpu_binary_cost_z = stk::GpuVolume(size, stk::Type_Float4);

    _labels = stk::VolumeUChar(size, uint8_t{0});
    _gpu_labels = stk::GpuVolume(size, stk::Type_UChar);
}
void HybridGraphCutOptimizer::reset_unary_cost()
{
    // TODO: Just a temp solution until GpuVolume::fill()

    cudaExtent extent = make_cudaExtent(
        _gpu_unary_cost.size().x * sizeof(float2), 
        _gpu_unary_cost.size().y, 
        _gpu_unary_cost.size().z
    );
    CUDA_CHECK_ERRORS(cudaMemset3D(_gpu_unary_cost.pitched_ptr(), 0, extent));
}

void HybridGraphCutOptimizer::enqueue_block(const Block& block)
{
    _cost_queue.push_back(block);
}
size_t HybridGraphCutOptimizer::dispatch_blocks(
    GpuUnaryFunction& unary_fn,
    GpuBinaryFunction& binary_fn,
    const float3& delta,
    double energy_epsilon,
    stk::GpuVolume& df
)
{
    reset_unary_cost();

    _labels.fill(0);
    
    while(!_cost_queue.empty()) {
        stk::cuda::Stream& stream = stk::cuda::Stream::null();

        Block block = _cost_queue.front();
        _cost_queue.pop_front();

        stream.synchronize();

        int3 block_dims = block.end - block.begin;

        // Compute unary terms for block
        unary_fn(df, delta, block.begin, block_dims, _gpu_unary_cost, stream);
        
        // Download the unary terms for the block into the large unary term volume. 
        download_subvolume(
            _gpu_unary_cost, 
            _unary_cost,
            block,
            false, // No padding for unary term
            stream
        );

        // Compute binary terms
        binary_fn(
            df,
            delta,
            block.begin,
            block_dims,
            _gpu_binary_cost_x,
            _gpu_binary_cost_y,
            _gpu_binary_cost_z,
            stream
        );

        // Download binary terms, since we're dependent on neighbouring terms at
        //  block borders we make sure to download them as well by padding
        //  negative directions by 1. This shouldn't affect the end results since
        //  these blocks are disabled anyways because of the red-black ordering.
        
        download_subvolume(
            _gpu_binary_cost_x, 
            _binary_cost_x,
            block,
            true,
            stream
        );

        download_subvolume(
            _gpu_binary_cost_y, 
            _binary_cost_y,
            block,
            true,
            stream
        );

        download_subvolume(
            _gpu_binary_cost_z, 
            _binary_cost_z,
            block,
            true,
            stream
        );

        stream.synchronize();
        _minimize_queue.push_back(block);
    }

    size_t num_blocks_changed = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:num_blocks_changed)
    for (int i = 0; i < _minimize_queue.size(); ++i) {
        Block block = _minimize_queue[i];
        //_minimize_queue.pop_front();

        bool changed = minimize_block(block, energy_epsilon);
        num_blocks_changed += changed ? 1 : 0;
    }
    _minimize_queue.clear();

    _gpu_labels.upload(_labels);

    apply_displacement_delta(delta, df, stk::cuda::Stream::null());

    return num_blocks_changed;
}
bool HybridGraphCutOptimizer::minimize_block(const Block& block, double energy_epsilon)
{
    dim3 full_dims = _labels.size();

    int3 block_dims {
        block.end.x - block.begin.x,
        block.end.y - block.begin.y,
        block.end.z - block.begin.z
    };

    GraphCut<double> graph(block_dims);

    double current_energy = 0;
    {
        PROFILER_SCOPE("build", 0xFF228844);
        
        for (int sub_z = 0; sub_z < block_dims.z; ++sub_z) {
            for (int sub_y = 0; sub_y < block_dims.y; ++sub_y) {
                for (int sub_x = 0; sub_x < block_dims.x; ++sub_x) {
                    // Global coordinates
                    int gx = block.begin.x + sub_x;
                    int gy = block.begin.y + sub_y;
                    int gz = block.begin.z + sub_z;

                    // Skip voxels outside volume
                    if (gx < 0 || gx >= int(full_dims.x) ||
                        gy < 0 || gy >= int(full_dims.y) ||
                        gz < 0 || gz >= int(full_dims.z)) {
                        graph.add_term1(sub_x, sub_y, sub_z, 0, 0);
                        continue;
                    }

                    double f0 = _unary_cost(gx, gy, gz).x;
                    double f1 = _unary_cost(gx, gy, gz).y;

                    // Block borders (excl image borders) (T-weights with binary term for neighboring voxels)

                    if (sub_x == 0 && gx != 0) {
                        f0 += _binary_cost_x(gx-1,gy,gz).x;
                        f1 += _binary_cost_x(gx-1,gy,gz).y;
                    }
                    else if (sub_x == block_dims.x - 1 && gx < int(full_dims.x) - 1) {
                        f0 += _binary_cost_x(gx,gy,gz).x;
                        f1 += _binary_cost_x(gx,gy,gz).z;
                    }

                    if (sub_y == 0 && gy != 0) {
                        f0 += _binary_cost_y(gx,gy-1,gz).x;
                        f1 += _binary_cost_y(gx,gy-1,gz).y;
                    }
                    else if (sub_y == block_dims.y - 1 && gy < int(full_dims.y) - 1) {
                        f0 += _binary_cost_y(gx,gy,gz).x;
                        f1 += _binary_cost_y(gx,gy,gz).z;
                    }

                    if (sub_z == 0 && gz != 0) {
                        f0 += _binary_cost_z(gx,gy,gz-1).x;
                        f1 += _binary_cost_z(gx,gy,gz-1).y;
                    }
                    else if (sub_z == block_dims.z - 1 && gz < int(full_dims.z) - 1) {
                        f0 += _binary_cost_z(gx,gy,gz).x;
                        f1 += _binary_cost_z(gx,gy,gz).z;
                    }

                    graph.add_term1(sub_x, sub_y, sub_z, f0, f1);

                    current_energy += f0;

                    if (sub_x + 1 < block_dims.x && gx + 1 < int(full_dims.x)) {
                        double f_same = _binary_cost_x(gx,gy,gz).x;
                        double f01 = _binary_cost_x(gx,gy,gz).y;
                        double f10 = _binary_cost_x(gx,gy,gz).z;

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x + 1, sub_y, sub_z, 
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                    if (sub_y + 1 < block_dims.y && gy + 1 < int(full_dims.y)) {
                        double f_same = _binary_cost_y(gx,gy,gz).x;
                        double f01 = _binary_cost_y(gx,gy,gz).y;
                        double f10 = _binary_cost_y(gx,gy,gz).z;

                        graph.add_term2(
                            sub_x, sub_y, sub_z,
                            sub_x, sub_y + 1, sub_z, 
                            f_same, f01, f10, f_same);

                        current_energy += f_same;
                    }
                    if (sub_z + 1 < block_dims.z && gz + 1 < int(full_dims.z)) {
                        double f_same = _binary_cost_z(gx,gy,gz).x;
                        double f01 = _binary_cost_z(gx,gy,gz).y;
                        double f10 = _binary_cost_z(gx,gy,gz).z;

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


    double current_emin;
    {
        PROFILER_SCOPE("minimize", 0xFF985423);
        current_emin = graph.minimize();
    }

    bool changed_flag = false;

    if (1.0 - current_emin / current_energy > energy_epsilon) // Accept solution
    {
        PROFILER_SCOPE("apply", 0xFF767323);
        for (int sub_z = 0; sub_z < block_dims.z; sub_z++) {
            for (int sub_y = 0; sub_y < block_dims.y; sub_y++) {
                for (int sub_x = 0; sub_x < block_dims.x; sub_x++) {
                    // Global coordinates
                    int gx = block.begin.x + sub_x;
                    int gy = block.begin.y + sub_y;
                    int gz = block.begin.z + sub_z;

                    // Skip voxels outside volume
                    if (gx < 0 || gx >= int(full_dims.x) ||
                        gy < 0 || gy >= int(full_dims.y) ||
                        gz < 0 || gz >= int(full_dims.z))
                    {
                        continue;
                    }

                    _labels(gx,gy,gz) = (uint8_t)graph.get_var(sub_x, sub_y, sub_z);
                    if (_labels(gx,gy,gz) == 1) {
                        changed_flag = true;
                    }
                }
            }
        }
    }
    return changed_flag;
}
void HybridGraphCutOptimizer::download_subvolume(
    const stk::GpuVolume& src, 
    stk::Volume& tgt, 
    const Block& block,
    bool pad, // Pad all axes by 1 in negative direction for binary cost
    stk::cuda::Stream& stream)
{
    ASSERT(src.size() == tgt.size());

    int3 padded_begin = block.begin;

    if (pad) {
        if (padded_begin.x > 0) padded_begin.x -= 1;
        if (padded_begin.y > 0) padded_begin.y -= 1;
        if (padded_begin.z > 0) padded_begin.z -= 1;
    }

    stk::GpuVolume sub_src(src,
        { padded_begin.x, block.end.x },
        { padded_begin.y, block.end.y },
        { padded_begin.z, block.end.z }
    );

    stk::Volume sub_tgt(tgt,
        { padded_begin.x, block.end.x },
        { padded_begin.y, block.end.y },
        { padded_begin.z, block.end.z }
    );

    sub_src.download(sub_tgt, stream);
}
