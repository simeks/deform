#include "binary_function.h"

namespace cuda = stk::cuda;

inline float phi(
    const float4& dv,
    const float4& dw,
    const float4& delta,
    float scale,
    float half_exponent,
    int lv,
    int lw) {
    return pow(scale * stk::norm2(dv + lv * delta - dw - lw * delta), half_exponent);
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
        for w in an intermediate buffer in shared memory. The kernel will add these capacities 
        at the end.

        [Kolmogorov et al. "What Energy Functions Can Be Minimized via Graph Cuts?"]
    */

    extern __shared__ float shared[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= dims.x ||
        y >= dims.y ||
        z >= dims.z) {
        return;
    }

    int shared_idx = x + y * blockDim.x + z * blockDim.x * blockDim.y;
    shared[shared_idx] = 0;
    
    int vx = x + offset.x;
    int vy = y + offset.y;
    int vz = z + offset.z;

    int wx = vx + step.x;
    int wy = vy + step.y;
    int wz = vz + step.z;

    if (wx >= df_dims.x ||
        wy >= df_dims.y ||
        wz >= df_dims.z) {
        return;
    }
    
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
        shared 
        cap_sink(vx, vy, vz) += e01;
        shared[shared_idx] += -e01;

        cap_edge_l(vx, vy, vz) = 0;
        cap_edge_r(vx, vy, vz) = e01 + e10;
    }
    else if (e10 < 0) {
        cap_sink(vx, vy, vz) += -e10;
        shared[shared_idx] += e10;

        cap_edge_l(vx, vy, vz) = e01 + e10;
        cap_edge_r(vx, vy, vz) = 0;
    }
    else {
        cap_edge_l(vx, vy, vz) = e01;
        cap_edge_r(vx, vy, vz) = e10;
    }
    __syncthreads();

    cap_sink(wx, wy, wz) += shared[shared_idx];
}


        // // Compute terminal capacities at block border
        // if (x == 0 && gx != 0) {
        //     float4 du = df(vx-step.x, vy-step.y, vz-step.z) - 
        //                 initial_df(vx-step.x, vy-step.y, vz-step.z);
            
        //     float e00 = weight*phi(dv, du, delta4, scale, half_exponent, 0, 0);
        //     float e01 = weight*phi(dv, du, delta4, scale, half_exponent, 0, 1);

        //     cap_src(vx, vy, vz) += e00;
        //     cap_sink(vx, vy, vz) += e01;
        // }
        // else if (step.x == 1 && x == dims.x - 1) {
        //     float4 du = df(vx-1, vy, vz) - initial_df(vx-1, vy, vz);
            
        //     float e00 = weight*phi(dv, du, delta4, scale, half_exponent, 0, 0);
        //     float e01 = weight*phi(dv, du, delta4, scale, half_exponent, 0, 1);

        //     cap_src(vx, vy, vz) += e00;
        //     cap_sink(vx, vy, vz) += e01;
        // }

        // if (step.x == 1 && x == 0 && gx != 0) {
        //     float4 du = df(vx-1, vy, vz) - initial_df(vx-1, vy, vz);
            
        //     float e00 = weight*phi(dv, du, delta4, scale, half_exponent, 0, 0);
        //     float e01 = weight*phi(dv, du, delta4, scale, half_exponent, 0, 1);

        //     cap_src(vx, vy, vz) += e00;
        //     cap_sink(vx, vy, vz) += e01;
        // }
        // else if (step.x == 1 && x == dims.x - 1) {
        //     float4 du = df(vx-1, vy, vz) - initial_df(vx-1, vy, vz);
            
        //     float e00 = weight*phi(dv, du, delta4, scale, half_exponent, 0, 0);
        //     float e01 = weight*phi(dv, du, delta4, scale, half_exponent, 0, 1);

        //     cap_src(vx, vy, vz) += e00;
        //     cap_sink(vx, vy, vz) += e01;
        // }
     
        // if (sub_z == 0 && gz != 0) {
        //     f0 += _binary_cost_z(gx,gy,gz-1).x;
        //     f1 += _binary_cost_z(gx,gy,gz-1).y;
        // }
        // else if (sub_z == block_dims.z - 1 && gz < int(full_dims.z) - 1) {
        //     f0 += _binary_cost_z(gx,gy,gz).x;
        //     f1 += _binary_cost_z(gx,gy,gz).z;
        // }


     // Compute cost at block border

    if (x == 0 && gx != 0) {
        float4 dx = df(gx-1, gy, gz) - initial_df(gx-1, gy, gz);

        float4 diff_00 = d - dx;
        float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        dist2_00 = pow(scale * dist2_00, half_exponent);

        float4 diff_01 = (d+delta4) - dx;
        float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        dist2_01 = pow(scale * dist2_01, half_exponent);

        cost_x(gx-1,gy,gz).x = weight*inv_spacing2_exp.x*dist2_00;
        cost_x(gx-1,gy,gz).y = weight*inv_spacing2_exp.x*dist2_01;
        cost_x(gx-1,gy,gz).z = weight*inv_spacing2_exp.x*dist2_00; // border nodes can't move
        cost_x(gx-1,gy,gz).w = cost_x(gx-1,gy,gz).x;
    }

    if (y == 0 && gy != 0) {
        float4 dy = df(gx, gy-1, gz) - initial_df(gx, gy-1, gz);

        float4 diff_00 = d - dy;
        float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        dist2_00 = pow(scale * dist2_00, half_exponent);

        float4 diff_01 = (d+delta4) - dy;
        float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        dist2_01 = pow(scale * dist2_01, half_exponent);

        cost_y(gx,gy-1,gz).x = weight*inv_spacing2_exp.y*dist2_00;
        cost_y(gx,gy-1,gz).y = weight*inv_spacing2_exp.y*dist2_01;
        cost_y(gx,gy-1,gz).z = weight*inv_spacing2_exp.y*dist2_00; // border nodes can't move
        cost_y(gx,gy-1,gz).w = cost_x(gx,gy-1,gz).x;
    }

    if (z == 0 && gz != 0) {
        float4 dz = df(gx, gy, gz-1) - initial_df(gx, gy, gz-1);

        float4 diff_00 = d - dz;
        float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        dist2_00 = pow(scale * dist2_00, half_exponent);

        float4 diff_01 = (d+delta4) - dz;
        float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        dist2_01 = pow(scale * dist2_01, half_exponent);

        cost_z(gx,gy,gz-1).x = weight*inv_spacing2_exp.z*dist2_00;
        cost_z(gx,gy,gz-1).y = weight*inv_spacing2_exp.z*dist2_01;
        cost_z(gx,gy,gz-1).z = weight*inv_spacing2_exp.z*dist2_00; // border nodes can't move
        cost_z(gx,gy,gz-1).w = cost_x(gx,gy,gz-1).x;
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
    ASSERT(cost_x.voxel_type() == stk::Type_Float);
    ASSERT(cost_y.voxel_type() == stk::Type_Float);
    ASSERT(cost_z.voxel_type() == stk::Type_Float);


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

    size_t shared_size = block_size.x*block_size.y*block_size.z*sizeof(float);
    regularizer_kernel_step<<<grid_size, block_size, shared_size, stream>>>(
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
    regularizer_kernel_step<<<grid_size, block_size, shared_size, stream>>>(
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
    regularizer_kernel_step<<<grid_size, block_size, shared_size, stream>>>(
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
    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

