#include "block_change_flags.h"
#include "gpu/cost_functions/cost_function.h"
#include "hybrid_graph_cut_optimizer.h"
#include "worker_pool.h"

#include "deform_lib/graph_cut/graph_cut.h"
#include "deform_lib/profiler/profiler.h"

#include <stk/cuda/stream.h>
#include <stk/image/gpu_volume.h>

#include <iomanip>

namespace {
    const int _neighbor_count = 6;
    int3 _neighbors[] = {
        {1, 0, 0},
        {-1, 0, 0},
        {0, 1, 0},
        {0, -1, 0},
        {0, 0, 1},
        {0, 0, -1}
    };
}

HybridGraphCutOptimizer::HybridGraphCutOptimizer(
    const Settings::Level& settings,
    GpuUnaryFunction& unary_fn,
    GpuBinaryFunction& binary_fn,
    stk::GpuVolume& df,
    WorkerPool& worker_pool,
    std::vector<stk::cuda::Stream>& stream_pool) :
    _settings(settings),
    _worker_pool(worker_pool),
    _stream_pool(stream_pool),
    _unary_fn(unary_fn),
    _binary_fn(binary_fn),
    _df(df),
    _current_delta{0, 0, 0}
{
    allocate_cost_buffers(df.size());
}
HybridGraphCutOptimizer::~HybridGraphCutOptimizer()
{
}

void HybridGraphCutOptimizer::execute()
{
    PROFILER_SCOPE("execute", 0xFF532439);

    dim3 dims = _df.size();

    // Setting the block size to (0, 0, 0) will disable blocking and run the whole volume
    int3 block_dims = _settings.block_size;
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

    _block_change_flags = BlockChangeFlags(block_count);

    int num_iterations = 0;
    LOG(Info) << "Initial Energy: " << calculate_energy();

    bool done = false;
    while (!done) {
        PROFILER_SCOPE("iteration", 0xFF39842A);

        // A max_iteration_count of -1 means we run until we converge
        if (_settings.max_iteration_count != -1 && num_iterations >= _settings.max_iteration_count)
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

                if (block_count.x > 1 && (block_count.x * block_dims.x - block_offset.x) < (int)dims.x)
                    real_block_count.x += 1;
                if (block_count.y > 1 && (block_count.y * block_dims.y - block_offset.y) < (int)dims.y)
                    real_block_count.y += 1;
                if (block_count.z > 1 && (block_count.z * block_dims.z - block_offset.z) < (int)dims.z)
                    real_block_count.z += 1;
            }

            // Only do red-black when having more than 1 block
            int num_blocks = real_block_count.x * real_block_count.y * real_block_count.z;
            for (int black_or_red = 0; black_or_red < (num_blocks > 1 ? 2 : 1); black_or_red++) {
                PROFILER_SCOPE("red_black", 0xFF339955);

                for (int n = 0; n < _neighbor_count; ++n) {
                    PROFILER_SCOPE("step", 0xFFAA6FE2);

                    // delta in [mm]
                    _current_delta = {
                        _settings.step_size.x * _neighbors[n].x,
                        _settings.step_size.y * _neighbors[n].y,
                        _settings.step_size.z * _neighbors[n].z
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

                        bool need_update = _block_change_flags.is_block_set(block_idx, use_shift == 1);
                        for (int i = 0; i < _neighbor_count; ++i) {
                            int3 neighbor = block_idx + _neighbors[i];
                            if (0 <= neighbor.x && neighbor.x < real_block_count.x &&
                                0 <= neighbor.y && neighbor.y < real_block_count.y &&
                                0 <= neighbor.z && neighbor.z < real_block_count.z) {
                                need_update = need_update || _block_change_flags.is_block_set(neighbor, use_shift == 1);
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
                        block.idx = block_idx;
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
                        block.shift = use_shift == 1;

                        _block_queue.push_back(block);
                    }
                    num_blocks_changed += dispatch_blocks();
                }
            }
        }

        done = num_blocks_changed == 0;

        LOG(Verbose) << "Iteration " << num_iterations << ", "
                     << "Changed " << num_blocks_changed << " blocks, "
                     << "Energy " << std::fixed << std::setprecision(9)
                                  << calculate_energy();

        ++num_iterations;

        PROFILER_FLIP();
    }
    LOG(Info) << "Energy: " << calculate_energy() << ", "
              << "Iterations: " << num_iterations;
}
void HybridGraphCutOptimizer::allocate_cost_buffers(const dim3& size)
{
    _cap_source = stk::Volume(size, stk::Type_Float, nullptr, stk::Usage_Pinned);
    _cap_source.fill({});
    _cap_sink = stk::Volume(size, stk::Type_Float, nullptr, stk::Usage_Pinned);
    _cap_sink.fill({});

    _cap_lee = stk::Volume(size, stk::Type_Float, nullptr, stk::Usage_Pinned);
    _cap_lee.fill({});
    _cap_gee = stk::Volume(size, stk::Type_Float, nullptr, stk::Usage_Pinned);
    _cap_gee.fill({});

    _cap_ele = stk::Volume(size, stk::Type_Float, nullptr, stk::Usage_Pinned);
    _cap_ele.fill({});
    _cap_ege = stk::Volume(size, stk::Type_Float, nullptr, stk::Usage_Pinned);
    _cap_ege.fill({});

    _cap_eel = stk::Volume(size, stk::Type_Float, nullptr, stk::Usage_Pinned);
    _cap_eel.fill({});
    _cap_eeg = stk::Volume(size, stk::Type_Float, nullptr, stk::Usage_Pinned);
    _cap_eeg.fill({});

    _gpu_cap_source = stk::GpuVolume(size, stk::Type_Float);
    _gpu_cap_sink = stk::GpuVolume(size, stk::Type_Float);

    _gpu_cap_lee = stk::GpuVolume(size, stk::Type_Float);
    _gpu_cap_gee = stk::GpuVolume(size, stk::Type_Float);
    _gpu_cap_ele = stk::GpuVolume(size, stk::Type_Float);
    _gpu_cap_ege = stk::GpuVolume(size, stk::Type_Float);
    _gpu_cap_eel = stk::GpuVolume(size, stk::Type_Float);
    _gpu_cap_eeg = stk::GpuVolume(size, stk::Type_Float);

    _labels = stk::Volume(size, stk::Type_UChar, nullptr, stk::Usage_Pinned);
    _gpu_labels = stk::GpuVolume(size, stk::Type_UChar);
}
void HybridGraphCutOptimizer::reset_terminal_capacities()
{
    // TODO: Just a temp solution until GpuVolume::fill()

    cudaExtent extent = make_cudaExtent(
        _gpu_cap_source.size().x * sizeof(float),
        _gpu_cap_source.size().y,
        _gpu_cap_source.size().z
    );
    CUDA_CHECK_ERRORS(cudaMemset3D(_gpu_cap_source.pitched_ptr(), 0, extent));
    CUDA_CHECK_ERRORS(cudaMemset3D(_gpu_cap_sink.pitched_ptr(), 0, extent));
}

size_t HybridGraphCutOptimizer::dispatch_blocks()
{
    // Terminal capacity volumes are used as accumulators, compared to edge capacities which are not.
    //  Therefore we need to reset them between each iteration.
    reset_terminal_capacities();

    // Reset labels
    _labels.fill(0);

    // Reset change count
    _num_blocks_changed = 0;
    _num_blocks_remaining = _block_queue.size();

    // Reserve first stream for uploading labels
    for (int i = 1; i < (int) _stream_pool.size(); ++i) {
        stk::cuda::Stream stream = _stream_pool[i];
        std::scoped_lock lock(_block_queue_lock);
        if (!_block_queue.empty()) {
            Block block = _block_queue.front();
            _block_queue.pop_front();

            _worker_pool.push_back([this, block, stream](){
                this->block_cost_task(block, stream);
            });
        }
    }

    while (_num_blocks_remaining > 0) {
        // Attempt to assist worker pool
        auto task = _worker_pool.try_pop();
        if (task)
            (*task)();
        else
            std::this_thread::yield();
    }

    {
        PROFILER_SCOPE("apply", 0xFF532439);
        _gpu_labels.upload(_labels);
        apply_displacement_delta(stk::cuda::Stream::null());
    }

    return _num_blocks_changed;
}
void HybridGraphCutOptimizer::dispatch_next_cost_block(stk::cuda::Stream stream)
{
    // Dispatch next cost block (if any)
    std::scoped_lock lock(_block_queue_lock);

    if (!_block_queue.empty()) {
        Block block = _block_queue.front();
        _block_queue.pop_front();

        // Cost computation should take higher priority since the minimization is
        //  dependent on the result, so we push to the front of the queue.
        _worker_pool.push_front([this, block, stream](){
            this->block_cost_task(block, stream);
        });
    }
}
void HybridGraphCutOptimizer::dispatch_minimize_block(const Block& block)
{
    _worker_pool.push_back([this, block](){
        this->minimize_block_task(block);
        --this->_num_blocks_remaining;
    });
}
void HybridGraphCutOptimizer::download_subvolume(
    const stk::GpuVolume& src,
    stk::Volume& tgt,
    const Block& block,
    bool pad, // Pad all axes by 1 in negative direction for binary cost
    stk::cuda::Stream stream)
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
void HybridGraphCutOptimizer::upload_subvolume(
    const stk::Volume& src,
    stk::GpuVolume& tgt,
    const Block& block,
    stk::cuda::Stream stream
)
{
    ASSERT(src.size() == tgt.size());

    stk::Volume sub_src(src,
        { block.begin.x, block.end.x },
        { block.begin.y, block.end.y },
        { block.begin.z, block.end.z }
    );

    stk::GpuVolume sub_tgt(tgt,
        { block.begin.x, block.end.x },
        { block.begin.y, block.end.y },
        { block.begin.z, block.end.z }
    );

    sub_tgt.upload(sub_src, stream);
}
void HybridGraphCutOptimizer::block_cost_task(
    const Block& block,
    stk::cuda::Stream stream
)
{
    PROFILER_SCOPE("block_cost", 0xFF532439);

    int3 block_dims = block.end - block.begin;

    // Compute unary terms for block
    _unary_fn(
        _df,
        _current_delta,
        block.begin,
        block_dims,
        _gpu_cap_source,
        _gpu_cap_sink,
        stream);

    // Compute binary terms
    _binary_fn(
        _df,
        _current_delta,
        block.begin,
        block_dims,
        _gpu_cap_source,
        _gpu_cap_sink,
        _gpu_cap_lee,
        _gpu_cap_gee,
        _gpu_cap_ele,
        _gpu_cap_ege,
        _gpu_cap_eel,
        _gpu_cap_eeg,
        stream
    );

    // Download the terminal capacities for the block into the large unary term volume.
    download_subvolume(
        _gpu_cap_source,
        _cap_source,
        block,
        false, // No padding for unary term
        stream
    );
    download_subvolume(
        _gpu_cap_sink,
        _cap_sink,
        block,
        false, // No padding for unary term
        stream
    );

    // Download edge capacities, since we're dependent on neighbouring terms at
    //  block borders we make sure to download them as well by padding
    //  negative directions by 1. This shouldn't affect the end results since
    //  these blocks are disabled anyways because of the red-black ordering.

    download_subvolume(
        _gpu_cap_lee,
        _cap_lee,
        block,
        true,
        stream
    );

    download_subvolume(
        _gpu_cap_gee,
        _cap_gee,
        block,
        true,
        stream
    );

    download_subvolume(
        _gpu_cap_ele,
        _cap_ele,
        block,
        true,
        stream
    );
    download_subvolume(
        _gpu_cap_ege,
        _cap_ege,
        block,
        true,
        stream
    );

    download_subvolume(
        _gpu_cap_eel,
        _cap_eel,
        block,
        true,
        stream
    );

    download_subvolume(
        _gpu_cap_eeg,
        _cap_eeg,
        block,
        true,
        stream
    );

    // No queueing here, queuing next blocks will be performed in the callback
    //  invoked when the cost computation is completed. This will leave room
    //  for the work pool to work on other tasks

    stream.add_callback([this, block](stk::cuda::Stream stream, int){
        this->dispatch_next_cost_block(stream);
        this->dispatch_minimize_block(block);
    });
}

void HybridGraphCutOptimizer::minimize_block_task(const Block& block)
{
    PROFILER_SCOPE("minimize_block", 0xFF228844);

    dim3 full_dims = _labels.size();

    int3 block_dims {
        block.end.x - block.begin.x,
        block.end.y - block.begin.y,
        block.end.z - block.begin.z
    };

    //GraphCut<double> graph(block_dims);
    gco::Graph<double, double, double> graph(
        block_dims.x * block_dims.y * block_dims.z,
        block_dims.x * block_dims.y * block_dims.z * 3
    );
    graph.add_node(block_dims.x * block_dims.y * block_dims.z);

    auto get_index = [&](int x, int y, int z){
        return x + y*block_dims.x + z*block_dims.x*block_dims.y;
    };

    // TODO: What about this?
    // double current_energy = 0;
    
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
            graph.add_tweights(get_index(sub_x, sub_y, sub_z), 0, 0);
            continue;
        }

        double f0 = _cap_source(gx, gy, gz);
        double f1 = _cap_sink(gx, gy, gz);

        graph.add_tweights(get_index(sub_x, sub_y, sub_z), f0, f1);

        
        if (sub_x + 1 < block_dims.x && gx + 1 < int(full_dims.x)) {
            graph.add_edge(
                get_index(sub_x, sub_y, sub_z),
                get_index(sub_x + 1, sub_y, sub_z),
                _cap_lee(gx, gy, gz),
                _cap_gee(gx, gy, gz)
            );
        }
        if (sub_y + 1 < block_dims.y && gy + 1 < int(full_dims.y)) {
            graph.add_edge(
                get_index(sub_x, sub_y, sub_z),
                get_index(sub_x, sub_y + 1, sub_z),
                _cap_ele(gx, gy, gz),
                _cap_ege(gx, gy, gz)
            );
        }
        if (sub_z + 1 < block_dims.z && gz + 1 < int(full_dims.z)) {
            graph.add_edge(
                get_index(sub_x, sub_y, sub_z),
                get_index(sub_x, sub_y, sub_z+ 1),
                _cap_eel(gx, gy, gz),
                _cap_eeg(gx, gy, gz)
            );
        }
    }
    }
    }

    double current_emin = graph.maxflow();

    bool changed_flag = false;
    //if (1.0 - current_emin / current_energy > _settings.block_energy_epsilon) // Accept solution
    {
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
                gz < 0 || gz >= int(full_dims.z))
            {
                continue;
            }

            if (graph.what_segment(get_index(sub_x, sub_y, sub_z)) == 1) {
                _labels(gx,gy,gz) = 1;
                changed_flag = true;
            }
        }
        }
        }
    }
    if (changed_flag) {
        _block_change_flags.set_block(block.idx, changed_flag, block.shift);
        ++_num_blocks_changed;
    }
}
