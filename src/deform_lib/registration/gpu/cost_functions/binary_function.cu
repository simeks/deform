#include "binary_function.h"

#include <stk/math/float4.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/cuda/volume.h>

namespace cuda = stk::cuda;

__device__ float4 energy(
    float4 d0,
    float4 d1,
    float4 dn0,
    float4 dn1,
    float scale,
    float half_exponent)
{
    return {
        pow(scale * stk::norm2(d0 - dn0), half_exponent),
        pow(scale * stk::norm2(d0 - dn1), half_exponent),
        pow(scale * stk::norm2(d1 - dn0), half_exponent),
        pow(scale * stk::norm2(d1 - dn1), half_exponent)
    };
}

struct CompositiveUpdate
{
    __device__ float4 operator()(
        const cuda::VolumePtr<float4>& df,
        const dim3& dims,
        int x, int y, int z,
        const float4& delta
    ) {
        return cuda::linear_at_clamp<float4>(
            df,
            dims,
            x + delta.x,
            y + delta.y,
            z + delta.z
        ) + delta;
    }
};

struct AdditiveUpdate
{
    __device__ float4 operator()(
        const cuda::VolumePtr<float4>& df,
        const dim3& dims,
        int x, int y, int z,
        const float4& delta
    ) {
        return df(x, y, z) + delta;
    }
};


template<typename UpdateFn>
__global__ void regularizer_kernel(
    cuda::VolumePtr<float4> df,
    cuda::VolumePtr<float4> initial_df,
    float4 delta,
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
    UpdateFn update_fn;

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

    float4 d0 = df(gx, gy, gz) - initial_df(gx, gy, gz);
    float4 d1 = update_fn(df, df_dims, gx, gy, gz, delta) - initial_df(gx, gy, gz);

    float4 o_x = {0, 0, 0, 0};
    float4 o_y = {0, 0, 0, 0};
    float4 o_z = {0, 0, 0, 0};

    if (gx + 1 < (int) df_dims.x) {
        float4 dn0 = df(gx+1, gy, gz) - initial_df(gx+1, gy, gz);
        float4 dn1 = update_fn(df, df_dims, gx+1, gy, gz, delta) 
                        - initial_df(gx+1, gy, gz);

        o_x = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );
    }
    if (gy + 1 < (int) df_dims.y) {
        float4 dn0 = df(gx, gy+1, gz) - initial_df(gx, gy+1, gz);
        float4 dn1 = update_fn(df, df_dims, gx, gy+1, gz, delta) 
                        - initial_df(gx, gy+1, gz);

        o_y = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );
    }
    if (gz + 1 < (int) df_dims.z) {
        float4 dn0 = df(gx, gy, gz+1) - initial_df(gx, gy, gz+1);
        float4 dn1 = update_fn(df, df_dims, gx, gy, gz+1, delta) 
                        - initial_df(gx, gy, gz+1);

        o_z = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );
    }
    cost_x(gx,gy,gz) = weight*inv_spacing2_exp.x*o_x;
    cost_y(gx,gy,gz) = weight*inv_spacing2_exp.y*o_y;
    cost_z(gx,gy,gz) = weight*inv_spacing2_exp.z*o_z;


     // Compute cost at block border

    if (x == 0 && gx != 0) {
        float4 dn0 = df(gx-1, gy, gz) - initial_df(gx-1, gy, gz);
        float4 dn1 = update_fn(df, df_dims, gx-1, gy, gz, delta) 
                        - initial_df(gx-1, gy, gz);
        
        float4 e = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );

        cost_x(gx-1,gy,gz).x = weight*inv_spacing2_exp.x*e.x;
        cost_x(gx-1,gy,gz).y = weight*inv_spacing2_exp.x*e.y;
        cost_x(gx-1,gy,gz).z = weight*inv_spacing2_exp.x*e.x; // border nodes can't move
        cost_x(gx-1,gy,gz).w = cost_x(gx-1,gy,gz).x;
    }

    if (y == 0 && gy != 0) {
        float4 dn0 = df(gx, gy-1, gz) - initial_df(gx, gy-1, gz);
        float4 dn1 = update_fn(df, df_dims, gx, gy-1, gz, delta) 
                        - initial_df(gx, gy-1, gz);
        
        float4 e = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );

        cost_x(gx,gy-1,gz).x = weight*inv_spacing2_exp.x*e.x;
        cost_x(gx,gy-1,gz).y = weight*inv_spacing2_exp.x*e.y;
        cost_x(gx,gy-1,gz).z = weight*inv_spacing2_exp.x*e.x; // border nodes can't move
        cost_x(gx,gy-1,gz).w = cost_x(gx,gy-1,gz).x;
    }

    if (z == 0 && gz != 0) {
        float4 dn0 = df(gx, gy, gz-1) - initial_df(gx, gy, gz-1);
        float4 dn1 = update_fn(df, df_dims, gx, gy, gz-1, delta) 
                        - initial_df(gx, gy, gz-1);
        
        float4 e = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );

        cost_x(gx,gy,gz-1).x = weight*inv_spacing2_exp.x*e.x;
        cost_x(gx,gy,gz-1).y = weight*inv_spacing2_exp.x*e.y;
        cost_x(gx,gy,gz-1).z = weight*inv_spacing2_exp.x*e.x; // border nodes can't move
        cost_x(gx,gy,gz-1).w = cost_x(gx,gy,gz-1).x;
    }
}

void GpuBinaryFunction::operator()(
        const stk::GpuVolume& df,
        const float3& delta,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cost_x,
        stk::GpuVolume& cost_y,
        stk::GpuVolume& cost_z,
        Settings::UpdateRule update_rule,
        stk::cuda::Stream& stream
        )
{
    DASSERT(df.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(df.voxel_type() == stk::Type_Float4);
    ASSERT(cost_x.voxel_type() == stk::Type_Float4);
    ASSERT(cost_y.voxel_type() == stk::Type_Float4);
    ASSERT(cost_z.voxel_type() == stk::Type_Float4);

    // The binary function is quite register heavy so we need
    // to restrict the thread count (512 rather than 1024).
    dim3 block_size {32, 16, 1};
    if (dims.x <= 16 || dims.y <= 16) {
        block_size = {16, 16, 2};
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

    float4 delta4 {
        delta.x,
        delta.y,
        delta.z,
        0
    };

    if (update_rule == Settings::UpdateRule_Compositive) {
        regularizer_kernel<CompositiveUpdate>
        <<<grid_size, block_size, 0, stream>>>(
            df,
            _initial,
            delta4,
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
    }
    else if (update_rule == Settings::UpdateRule_Additive) {
        regularizer_kernel<AdditiveUpdate>
        <<<grid_size, block_size, 0, stream>>>(
            df,
            _initial,
            delta4,
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
    }
    else {
        ASSERT(false);
    }

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

