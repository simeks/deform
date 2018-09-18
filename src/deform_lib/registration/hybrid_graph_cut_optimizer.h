#pragma once

#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include "settings.h"

#include <deque>

class GpuBinaryFunction;
class GpuUnaryFunction;

// Hybrid (CPU-GPU) graph-cut optimizer. 
//  Uses GPU to compute weights for graph and minimizes graph on CPU.
class HybridGraphCutOptimizer
{
public:
    HybridGraphCutOptimizer();
    ~HybridGraphCutOptimizer();

    void execute(
        const Settings::Level& settings,
        GpuUnaryFunction& unary_fn,
        GpuBinaryFunction& binary_fn,
        stk::GpuVolume& df
    );

private:
    struct Block
    {
        int3 begin;
        int3 end;
    };

    // Allocates CPU and GPU buffers for the costs
    void allocate_cost_buffers(const dim3& size);

    // Sets the unary cost buffer to all zeros
    void reset_unary_cost();

    // Enqueues the given block to the pipeline queue
    void enqueue_block(const Block& block);

    // Dispatches all queues block
    // Returns the number of changed blocks
    size_t dispatch_blocks(
        GpuUnaryFunction& unary_fn,
        GpuBinaryFunction& binary_fn,
        const float3& delta,
        double energy_epsilon,
        stk::GpuVolume& df
    );

    // Performs graph cut on block
    // Returns true if block was changed
    bool minimize_block(const Block& block, double energy_epsilon);
    
    // Applies delta based on labels in _gpu_labels.
    void apply_displacement_delta(
        const float3& delta,
        stk::GpuVolume& df,
        stk::cuda::Stream& stream
    );

    void download_subvolume(
        const stk::GpuVolume& src, 
        stk::Volume& tgt, 
        const Block& block,
        bool pad, // Pad all axes by 1 in negative direction for binary cost
        stk::cuda::Stream& stream
    );

    // Calculates the energy sum for the given displacement field
    double calculate_energy(
        GpuUnaryFunction& unary_fn,
        GpuBinaryFunction& binary_fn,
        stk::GpuVolume& df
    );

    stk::VolumeFloat2 _unary_cost;
    stk::GpuVolume _gpu_unary_cost;
    
    stk::VolumeFloat4 _binary_cost_x;
    stk::VolumeFloat4 _binary_cost_y;
    stk::VolumeFloat4 _binary_cost_z;

    stk::GpuVolume _gpu_binary_cost_x;
    stk::GpuVolume _gpu_binary_cost_y;
    stk::GpuVolume _gpu_binary_cost_z;

    // Labels from the minimization
    stk::VolumeUChar _labels;
    stk::GpuVolume _gpu_labels;

    // Blocks awaiting cost computation
    std::deque<Block> _cost_queue;
    // Blocks awaiting minimization
    std::deque<Block> _minimize_queue;

    int3 _neighbors[6];
};
