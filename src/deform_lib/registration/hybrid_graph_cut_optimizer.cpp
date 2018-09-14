#include "hybrid_graph_cut_optimizer.h"

#include <stk/cuda/stream.h>
#include <stk/image/gpu_volume.h>

namespace {
    void download_subvolume(
        const stk::GpuVolume& src, 
        stk::Volume& tgt, 
        const int3& offset,
        const int3& dims,
        bool pad, // Pad all axes by 1 in negative direction for binary cost
        stk::cuda::Stream& stream)
    {
        ASSERT(src.size() == tgt.size());

        int3 padded_offset = offset;
        int3 padded_dims = dims;

        if (pad) {
            if (offset.x > 0) {
                padded_offset.x -= 1;
                padded_dims.x += 1;
            }
            if (offset.y > 0) {
                padded_offset.y -= 1;
                padded_dims.y += 1;
            }
            if (offset.z > 0) {
                padded_offset.z -= 1;
                padded_dims.z += 1;
            }
        }

        dim3 src_size = src.size();
        padded_offset.x = std::max<int>(0, std::min<int>(padded_offset.x, src_size.x));
        padded_offset.y = std::max<int>(0, std::min<int>(padded_offset.y, src_size.y));
        padded_offset.z = std::max<int>(0, std::min<int>(padded_offset.z, src_size.z));

        padded_dims.x = std::max<int>(0, std::min<int>(padded_dims.x, src_size.x - padded_offset.x));
        padded_dims.y = std::max<int>(0, std::min<int>(padded_dims.y, src_size.y - padded_offset.y));
        padded_dims.z = std::max<int>(0, std::min<int>(padded_dims.z, src_size.z - padded_offset.z));

        stk::GpuVolume sub_src(src,
            { padded_offset.x, padded_offset.x + padded_dims.x },
            { padded_offset.y, padded_offset.y + padded_dims.y },
            { padded_offset.z, padded_offset.z + padded_dims.z }
        );

        stk::Volume sub_tgt(tgt,
            { padded_offset.x, padded_offset.x + padded_dims.x },
            { padded_offset.y, padded_offset.y + padded_dims.y },
            { padded_offset.z, padded_offset.z + padded_dims.z }
        );

        sub_src.download(sub_tgt, stream);
    }
}

HybridGraphCutOptimizer::HybridGraphCutOptimizer()
{
}
HybridGraphCutOptimizer::~HybridGraphCutOptimizer()
{
}

void HybridGraphCutOptimizer::execute(
    const Settings::Level& settings,
    GpuUnaryFunction& unary_fn,
    GpuBinaryFunction& binary_fn,
    const float3& step_size,
    stk::GpuVolume& df
)
{
    dim3 dims = def.size();

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
                
                if (block_count.x > 1) real_block_count.x += 1;
                if (block_count.y > 1) real_block_count.y += 1;
                if (block_count.z > 1) real_block_count.z += 1;
            }
            
            for (int black_or_red = 0; black_or_red < 2; black_or_red++) {
                PROFILER_SCOPE("red_black", 0xFF339955);
                int num_blocks = real_block_count.x * real_block_count.y * real_block_count.z;
                
                const int n_count = 6; // Neighbors
                for (int n = 0; n < n_count; ++n) {
                    PROFILER_SCOPE("step", 0xFFAA6FE2);

                    // delta in [mm]
                    float3 delta {
                        step_size.x * _neighbors[n].x,
                        step_size.y * _neighbors[n].y,
                        step_size.z * _neighbors[n].z
                    };
                    
                    // Queue all blocks
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

                        enqueue_block({block_idx, block_dims, block_offset});
                    }
                    dispatch_blocks(delta);
                }
            }
        }

        done = num_blocks_changed == 0;

        ++num_iterations;

        // A max_iteration_count of -1 means we run until we converge
        if (_max_iteration_count != -1 && num_iterations >= _max_iteration_count)
            break;
        
        PROFILER_FLIP();
    }
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

    _labels = stk::VolumeUChar(size, 0);
    _gpu_labels = stk::GpuVolume(size, stk::Type_UChar);
}
void HybridGraphCutOptimizer::reset_unary_cost()
{
    // TODO: Just a temp solution until GpuVolume::fill()

    cudaExtent extent = make_cudaExtent(
        cost.size().x * sizeof(float2), 
        cost.size().y, 
        cost.size().z
    );
    CUDA_CHECK_ERRORS(cudaMemset3D(_gpu_unary_cost.pitched_ptr(), 0, extent));
}

void HybridGraphCutOptimizer::enqueue_block(const Block& block)
{
    _cost_queue.push_back(block);
}
void HybridGraphCutOptimizer::dispatch_blocks(
    GpuUnaryFunction& unary_fn,
    GpuBinaryFunction& binary_fn,
    const float3& delta,
    stk::GpuVolume& df
)
{
    _blocks_remaining = _gpu_queue.size();

    reset_unary_cost();

    _labels.fill(0);
    
    while(!_cost_queue.empty()) {
        stk::cuda::Stream& stream = stk::cuda::Stream::null();

        Block block = _cost_queue.front();
        _cost_queue.pop_front();

        int3 offset {
            block.idx.x * block.dims.x - block.offset.x;
            block.idx.y * block.dims.y - block.offset.y;
            block.idx.z * block.dims.z - block.offset.z;
        };

        stream.synchronize();

        // Compute unary terms for block
        unary_fn(offset, block.dims, delta, df, _gpu_unary_cost, stream);
        
        // Download the unary terms for the block into the large unary term volume. 
        download_subvolume(
            _gpu_unary_cost, 
            _unary_cost,
            offset,
            block.dims,
            false, // No padding for unary term
            stream
        );

        // Compute binary terms
        binary_fn(
            offset,
            block.dims,
            delta,
            df,
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
            {gx,gy,gz},
            block.dims,
            true,
            _streams[si].stream
        );

        download_subvolume(
            _gpu_binary_cost_y, 
            _binary_cost_y,
            {gx,gy,gz},
            block.dims,
            true,
            _streams[si].stream
        );

        download_subvolume(
            _gpu_binary_cost_z, 
            _binary_cost_z,
            int3{gx,gy,gz},
            block.dims,
            true,
            _streams[si].stream
        );

        stream.synchronize();
        _minimize_queue.push_back(block);
    }

    while (!_minimize_queue.empty()) {
        Block block = _minimize_queue.front();
        _minimize_queue.pop_front();


    }

}