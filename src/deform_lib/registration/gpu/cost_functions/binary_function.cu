#include "binary_function.h"

namespace cuda = stk::cuda;

__global__ void regularizer_kernel(
    cuda::VolumePtr<float4> df,
    cuda::VolumePtr<float4> initial_df,
    float3 delta,
    float weight,
    float scale,
    float half_exponent,
    int3 offset,
    int3 dims,
    dim3 df_dims,
    float3 inv_spacing2_exp,
    cuda::VolumePtr<float4> cost_x, // Regularization cost in x+
    cuda::VolumePtr<float4> cost_y, // y+
    cuda::VolumePtr<float4> cost_z  // z+
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

    int gx = x + offset.x;
    int gy = y + offset.y;
    int gz = z + offset.z;

    // Cost ordered as E00, E01, E10, E11

    float4 delta4 = {delta.x, delta.y, delta.z, 0.0f};
    float4 d = df(gx, gy, gz) - initial_df(gx, gy, gz);
    {
        float4 o_x = {0, 0, 0, 0};
        float4 o_y = {0, 0, 0, 0};
        float4 o_z = {0, 0, 0, 0};

        if (gx + 1 < (int) df_dims.x) {
            float4 dx = df(gx+1, gy, gz) - initial_df(gx+1, gy, gz);

            float4 diff_00 = d - dx;
            float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
            dist2_00 = pow(scale * dist2_00, half_exponent);

            float4 diff_01 = d - (dx+delta4);
            float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
            dist2_01 = pow(scale * dist2_01, half_exponent);

            float4 diff_10 = (d+delta4) - dx;
            float dist2_10 = diff_10.x*diff_10.x + diff_10.y*diff_10.y + diff_10.z*diff_10.z;
            dist2_10 = pow(scale * dist2_10, half_exponent);

            o_x.x = dist2_00;
            o_x.y = dist2_01;
            o_x.z = dist2_10;
            o_x.w = dist2_00; // E11 same as E00
            }
        if (gy + 1 < (int) df_dims.y) {
            float4 dy = df(gx, gy+1, gz) - initial_df(gx, gy+1, gz);

            float4 diff_00 = d - dy;
            float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
            dist2_00 = pow(scale * dist2_00, half_exponent);

            float4 diff_01 = d - (dy+delta4);
            float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
            dist2_01 = pow(scale * dist2_01, half_exponent);

            float4 diff_10 = (d+delta4) - dy;
            float dist2_10 = diff_10.x*diff_10.x + diff_10.y*diff_10.y + diff_10.z*diff_10.z;
            dist2_10 = pow(scale * dist2_10, half_exponent);

            o_y.x = dist2_00;
            o_y.y = dist2_01;
            o_y.z = dist2_10;
            o_y.w = dist2_00;
        }
        if (gz + 1 < (int) df_dims.z) {
            float4 dz = df(gx, gy, gz+1) - initial_df(gx, gy, gz+1);

            float4 diff_00 = d - dz;
            float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
            dist2_00 = pow(scale * dist2_00, half_exponent);

            float4 diff_01 = d - (dz+delta4);
            float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
            dist2_01 = pow(scale * dist2_01, half_exponent);

            float4 diff_10 = (d+delta4) - dz;
            float dist2_10 = diff_10.x*diff_10.x + diff_10.y*diff_10.y + diff_10.z*diff_10.z;
            dist2_10 = pow(scale * dist2_10, half_exponent);

            o_z.x = dist2_00;
            o_z.y = dist2_01;
            o_z.z = dist2_10;
            o_z.w = dist2_00;
        }
        cost_x(gx,gy,gz) = weight*inv_spacing2_exp.x*o_x;
        cost_y(gx,gy,gz) = weight*inv_spacing2_exp.y*o_y;
        cost_z(gx,gy,gz) = weight*inv_spacing2_exp.z*o_z;
    }

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
    ASSERT(cost_x.voxel_type() == stk::Type_Float4);
    ASSERT(cost_y.voxel_type() == stk::Type_Float4);
    ASSERT(cost_z.voxel_type() == stk::Type_Float4);

    dim3 block_size {32, 32, 1};

    if (dims.x <= 16 || dims.y <= 16) {
        block_size = {16, 16, 4};
    }

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    float3 inv_spacing2_exp {
        1.0f / pow(_spacing.x*_spacing.x, _half_exponent),
        1.0f / pow(_spacing.y*_spacing.y, _half_exponent),
        1.0f / pow(_spacing.z*_spacing.z, _half_exponent)
    };

    regularizer_kernel<<<grid_size, block_size, 0, stream>>>(
            df,
            _initial,
            delta,
            _weight,
            _scale,
            _half_exponent,
            offset,
            dims,
            df.size(),
            inv_spacing2_exp,
            cost_x,
            cost_y,
            cost_z
            );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

