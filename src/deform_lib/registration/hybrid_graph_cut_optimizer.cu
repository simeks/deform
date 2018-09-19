#include "gpu/cost_function.h"
#include "hybrid_graph_cut_optimizer.h"

#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/cuda/volume.h>

namespace cuda = stk::cuda;

__global__ void apply_displacement_delta_kernel(
    cuda::VolumePtr<float4> df,
    cuda::VolumePtr<uint8_t> labels,
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
    cuda::Stream& stream
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

__global__ void reduce_total_energy(
    cuda::VolumePtr<float2> unary_term,
    cuda::VolumePtr<float4> binary_term_x, // Regularization cost in x+
    cuda::VolumePtr<float4> binary_term_y, // y+
    cuda::VolumePtr<float4> binary_term_z, // z+
    dim3 dims,
    float* out
)
{
    extern __shared__ float shared[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    shared[tid] = 0;

    if (x < dims.x &&
        y < dims.y &&
        z < dims.z)
    {
        float e = unary_term(x,y,z).x;
        if (x + 1 < int(dims.x)) {
            e += binary_term_x(x,y,z).x;
        }
        if (y + 1 < int(dims.y)) {
            e += binary_term_y(x,y,z).x;
        }
        if (z + 1 < int(dims.z)) {
            e += binary_term_z(x,y,z).x;
        }

        shared[tid] = e;
    }
    __syncthreads();

    #define REDUCTION_STEP(n_) \
        if (tid < (n_)) shared[tid] = shared[tid] + shared[tid+(n_)]; \
        __syncthreads();

    REDUCTION_STEP(512);
    REDUCTION_STEP(256);
    REDUCTION_STEP(128);
    REDUCTION_STEP(64);
    REDUCTION_STEP(32);
    REDUCTION_STEP(16);
    REDUCTION_STEP(8);
    REDUCTION_STEP(4);
    REDUCTION_STEP(2);

    if (tid == 0) {
        out[bid] = shared[0] + shared[1];
    }

    #undef REDUCTION_STEP
}

double HybridGraphCutOptimizer::calculate_energy(
    GpuUnaryFunction& unary_fn,
    GpuBinaryFunction& binary_fn,
    stk::GpuVolume& df
)
{
    reset_unary_cost();

    dim3 dims = _gpu_unary_cost.size();
    int3 begin {0, 0, 0};
    int3 end {(int)dims.x, (int)dims.y, (int)dims.z};

    cuda::Stream& stream = stk::cuda::Stream::null();
    unary_fn(df, {0,0,0}, begin, end, _gpu_unary_cost, stream);
    
    // Compute binary terms
    binary_fn(
        df,
        {0, 0, 0},
        begin,
        end,
        _gpu_binary_cost_x,
        _gpu_binary_cost_y,
        _gpu_binary_cost_z,
        stream
    );

    dim3 block_size{32,32,1};

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };
    uint32_t n_blocks = grid_size.x * grid_size.y * grid_size.z;

    float* d_block_sum;
    CUDA_CHECK_ERRORS(cudaMalloc(&d_block_sum, n_blocks*sizeof(float)));

    reduce_total_energy<<<grid_size, block_size, 
        uint32_t(sizeof(float)*1024)>>>
    (
        _gpu_unary_cost,
        _gpu_binary_cost_x,
        _gpu_binary_cost_y,
        _gpu_binary_cost_z,
        dims,
        d_block_sum
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

    float* block_sum = new float[n_blocks];
    CUDA_CHECK_ERRORS(cudaMemcpy(block_sum, d_block_sum, n_blocks*sizeof(float), cudaMemcpyDeviceToHost));

    // TODO: Perform all reduction on GPU
    double total_energy = 0;
    for (int i = 0; i < (int)n_blocks; ++i) {
        total_energy += block_sum[i];
    }

    delete [] block_sum;
    CUDA_CHECK_ERRORS(cudaFree(d_block_sum));

    return total_energy;
}
