#include "hybrid_graph_cut_optimizer.h"

#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/cuda/volume.h>


__global__ void apply_displacement_delta_kernel(
    stk::cuda::VolumePtr<float4> df,
    stk::cuda::VolumePtr<uint8_t> labels,
    dim3 dims,
    float4 delta
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= dims.x ||
        y >= dims.y ||
        z >= dims.z)
    {
        return;
    }

    df(x,y,z) = df(x,y,z) + delta * labels(x,y,z);
}

void HybridGraphCutOptimizer::apply_displacement_delta(
    const float3& delta,
    stk::GpuVolume& df,
    stk::cuda::Stream& stream
)
{
    dim3 dims = df.size();
    dim3 block_size {32,32,1};
    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    apply_displacement_delta_kernel<<<grid_size, block_size, 0, stream>>>(
        df, 
        _gpu_labels, 
        dims, 
        float4{delta.x, delta.y, delta.z, 0.0f}
    );
    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}
