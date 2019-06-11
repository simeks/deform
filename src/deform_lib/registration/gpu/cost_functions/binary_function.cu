#include "binary_function.h"

#include <stk/math/float4.h>

namespace cuda = stk::cuda;

__device__ inline float phi(
    const float4& dv,
    const float4& dw,
    const float4& delta,
    float scale,
    float half_exponent,
    int lv,
    int lw) {
    return powf(scale * stk::norm2(dv + lv * delta - dw - lw * delta), half_exponent);
}

// Accumulate the terminal capacities
__global__ void regularizer_kernel_step(
    cuda::VolumePtr<float4> df,
    cuda::VolumePtr<float4> initial_df,
    int3 step,
    float3 delta,
    float weight,
    float scale,
    float half_exponent,
    int3 offset,
    int3 dims,
    dim3 df_dims,
    cuda::VolumePtr<float> cap_source,
    cuda::VolumePtr<float> cap_sink,
    cuda::VolumePtr<float> cap_edge_l,
    cuda::VolumePtr<float> cap_edge_g
)
{
    /*
        When converting the binary potential for v,w to edge capacities we have to update
        the terminal capacities (cap_source and cap_sink) for both v and w. For this reason
        we can't update both simultaneously. To solve this we store all terminal capacities
        for w in an intermediate buffer. The kernel will add these capacities at the end.

        [Kolmogorov et al. "What Energy Functions Can Be Minimized via Graph Cuts?"]
    */

    extern __shared__ float shared[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    // Skip edges spanning block borders
    if (x + step.x >= dims.x ||
        y + step.y >= dims.y ||
        z + step.z >= dims.z) {
        return;
    }

    float w_tmp = 0;
    
    int vx = x + offset.x;
    int vy = y + offset.y;
    int vz = z + offset.z;

    int wx = vx + step.x;
    int wy = vy + step.y;
    int wz = vz + step.z;

    float4 dv = df(vx, vy, vz) - initial_df(vx, vy, vz);
    float4 dw = df(wx, wy, wz) - initial_df(wx, wy, wz);
    
    float4 delta4 = {delta.x, delta.y, delta.z, 0.0f};

    float e00 = weight*phi(dv, dw, delta4, scale, half_exponent, 0, 0);
    float e01 = weight*phi(dv, dw, delta4, scale, half_exponent, 0, 1);
    float e10 = weight*phi(dv, dw, delta4, scale, half_exponent, 1, 0);
    float e11 = weight*phi(dv, dw, delta4, scale, half_exponent, 1, 1);

    cap_source(vx, vy, vz) += e00;
    cap_sink(vx, vy, vz) += e11;
    
    e01 -= e00; e10 -= e11;

    if (e01 < 0) {
        cap_sink(vx, vy, vz) += e01;
        w_tmp += -e01;

        cap_edge_l(vx, vy, vz) = 0;

        // Clamp to avoid < 0 due to rounding errors
        cap_edge_g(vx, vy, vz) = fmaxf(0, e01 + e10);
    }
    else if (e10 < 0) {
        cap_sink(vx, vy, vz) += -e10;
        w_tmp += e10;

        // Clamp to avoid < 0 due to rounding errors
        cap_edge_l(vx, vy, vz) = fmaxf(0, e01 + e10);
        cap_edge_g(vx, vy, vz) = 0;
    }
    else {
        cap_edge_l(vx, vy, vz) = e01;
        cap_edge_g(vx, vy, vz) = e10;
    }
    __syncthreads();

        // Nope.
    cap_sink(wx, wy, wz) += w_tmp;
}

__global__ void regularizer_kernel_borders_step(
    cuda::VolumePtr<float4> df,
    cuda::VolumePtr<float4> initial_df,
    int3 step,
    float3 delta,
    float weight,
    float scale,
    float half_exponent,
    int3 offset,
    int3 dims,
    dim3 df_dims,
    cuda::VolumePtr<float> cap_source,
    cuda::VolumePtr<float> cap_sink
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    // Skip edges spanning block borders
    if (x >= dims.x ||
        y >= dims.y ||
        z >= dims.z) {
        return;
    }

    int vx = x + offset.x;
    int vy = y + offset.y;
    int vz = z + offset.z;

    // First index for block in step axis
    int first = step.x * vx + step.y * vy + step.z * vz;
    // Last index for block in step axis
    int last = first + step.x * dims.x + step.y * dims.y + step.z * dims.z;
    
    float4 delta4 = {delta.x, delta.y, delta.z, 0.0f};
    float4 dv = df(vx, vy, vz) - initial_df(vx, vy, vz);
    
    // We don't want to add any edges to the outside of the volume
    if (first > 0) {
        // Add an edge to node outside block, assuming it has a static label

        float4 du = df(vx-step.x, vy-step.y, vz-step.z) - 
                    initial_df(vx-step.x, vy-step.y, vz-step.z);

        float e00 = weight*phi(dv, du, delta4, scale, half_exponent, 0, 0);
        float e10 = weight*phi(dv, du, delta4, scale, half_exponent, 1, 0);

        cap_source(vx, vy, vz) += e00;
        cap_sink(vx, vy, vz) += e10;
    }
    // Same goes here
    if (last < (df_dims.x * step.x + df_dims.y * step.y + df_dims.z * step.z)-1) {

        float4 dw = df(vx+step.x, vy+step.y, vz+step.z) - 
                    initial_df(vx+step.x, vy+step.y, vz+step.z);

        float e00 = weight*phi(dv, dw, delta4, scale, half_exponent, 0, 0);
        float e10 = weight*phi(dv, dw, delta4, scale, half_exponent, 1, 0);

        cap_source(vx, vy, vz) += e00;
        cap_sink(vx, vy, vz) += e10;
    }
}

void GpuBinaryFunction::operator()(
        const stk::GpuVolume& df,
        const float3& delta,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cap_source,
        stk::GpuVolume& cap_sink,
        stk::GpuVolume& cap_lee,
        stk::GpuVolume& cap_gee,
        stk::GpuVolume& cap_ele,
        stk::GpuVolume& cap_ege,
        stk::GpuVolume& cap_eel,
        stk::GpuVolume& cap_eeg,
        stk::cuda::Stream& stream
)
{
    DASSERT(df.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(df.voxel_type() == stk::Type_Float4);
    ASSERT(cap_source.voxel_type() == stk::Type_Float);
    ASSERT(cap_sink.voxel_type() == stk::Type_Float);
    ASSERT(cap_lee.voxel_type() == stk::Type_Float);
    ASSERT(cap_gee.voxel_type() == stk::Type_Float);
    ASSERT(cap_ele.voxel_type() == stk::Type_Float);
    ASSERT(cap_ege.voxel_type() == stk::Type_Float);
    ASSERT(cap_eel.voxel_type() == stk::Type_Float);
    ASSERT(cap_eeg.voxel_type() == stk::Type_Float);

    float3 inv_spacing2_exp {
        1.0f / pow(_spacing.x*_spacing.x, _half_exponent),
        1.0f / pow(_spacing.y*_spacing.y, _half_exponent),
        1.0f / pow(_spacing.z*_spacing.z, _half_exponent)
    };

    float3 weight{
        _weight * inv_spacing2_exp.x,
        _weight * inv_spacing2_exp.y,
        _weight * inv_spacing2_exp.z
    };

    dim3 block_size {8, 8, 8};

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    regularizer_kernel_step<<<grid_size, block_size, 0, stream>>>(
        df,
        _initial,
        int3{1, 0, 0},
        delta,
        weight.x,
        _scale,
        _half_exponent,
        offset, // block offset
        dims, // block size
        df.size(),
        cap_source,
        cap_sink,
        cap_lee,
        cap_gee
    );
    regularizer_kernel_step<<<grid_size, block_size, 0, stream>>>(
        df,
        _initial,
        int3{0, 1, 0},
        delta,
        weight.y,
        _scale,
        _half_exponent,
        offset, // block offset
        dims, // block size
        df.size(),
        cap_source,
        cap_sink,
        cap_ele,
        cap_ege
    );
    regularizer_kernel_step<<<grid_size, block_size, 0, stream>>>(
        df,
        _initial,
        int3{0, 0, 1},
        delta,
        weight.z,
        _scale,
        _half_exponent,
        offset, // block offset
        dims, // block size
        df.size(),
        cap_source,
        cap_sink,
        cap_eel,
        cap_eeg
    );

    // Borders

    block_size = {1,16,16};
    grid_size = {
        1,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    regularizer_kernel_borders_step<<<grid_size, block_size, 0, stream>>>(
        df,
        _initial,
        int3{1, 0, 0},
        delta,
        weight.x,
        _scale,
        _half_exponent,
        offset, // block offset
        dims, // block size
        df.size(),
        cap_source,
        cap_sink
    );

    block_size = {16,1,16};
    grid_size = {
        (dims.x + block_size.x - 1) / block_size.x,
        1,
        (dims.z + block_size.z - 1) / block_size.z
    };

    regularizer_kernel_borders_step<<<grid_size, block_size, 0, stream>>>(
        df,
        _initial,
        int3{0, 1, 0},
        delta,
        weight.y,
        _scale,
        _half_exponent,
        offset, // block offset
        dims, // block size
        df.size(),
        cap_source,
        cap_sink
    );

    block_size = {16,1,16};
    grid_size = {
        (dims.x + block_size.x - 1) / block_size.x,
        1,
        (dims.z + block_size.z - 1) / block_size.z
    };

    regularizer_kernel_borders_step<<<grid_size, block_size, 0, stream>>>(
        df,
        _initial,
        int3{0, 0, 1},
        delta,
        weight.z,
        _scale,
        _half_exponent,
        offset, // block offset
        dims, // block size
        df.size(),
        cap_source,
        cap_sink
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

