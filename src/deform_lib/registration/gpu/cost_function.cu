#include "cost_function.h"

#include <stk/common/assert.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/float4.h>

namespace cuda = stk::cuda;

__global__ void regularizer_kernel(
    cuda::VolumePtr<float4> df,
    cuda::VolumePtr<float4> initial_df,
    float3 delta,
    float weight,
    int3 offset,
    int3 dims,
    dim3 df_dims,
    float3 inv_spacing2,
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

        if (gx + 1 < df_dims.x) {
            float4 dx = df(gx+1, gy, gz) - initial_df(gx+1, gy, gz);

            float4 diff_00 = d - dx;
            float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
            
            float4 diff_01 = d - (dx+delta4);
            float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
            
            float4 diff_10 = (d+delta4) - dx;
            float dist2_10 = diff_10.x*diff_10.x + diff_10.y*diff_10.y + diff_10.z*diff_10.z;
            
            o_x.x = dist2_00;
            o_x.y = dist2_01;
            o_x.z = dist2_10;
            o_x.w = dist2_00; // E11 same as E00
            }
        if (gy + 1 < df_dims.y) {
            float4 dy = df(gx, gy+1, gz) - initial_df(gx, gy+1, gz);

            float4 diff_00 = d - dy;
            float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
            
            float4 diff_01 = d - (dy+delta4);
            float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
            
            float4 diff_10 = (d+delta4) - dy;
            float dist2_10 = diff_10.x*diff_10.x + diff_10.y*diff_10.y + diff_10.z*diff_10.z;

            o_y.x = dist2_00;
            o_y.y = dist2_01;
            o_y.z = dist2_10;
            o_y.w = dist2_00;
        }
        if (gz + 1 < df_dims.z) {
            float4 dz = df(gx, gy, gz+1) - initial_df(gx, gy, gz+1);

            float4 diff_00 = d - dz;
            float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
            
            float4 diff_01 = d - (dz+delta4);
            float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
            
            float4 diff_10 = (d+delta4) - dz;
            float dist2_10 = diff_10.x*diff_10.x + diff_10.y*diff_10.y + diff_10.z*diff_10.z;
            
            o_z.x = dist2_00;
            o_z.y = dist2_01;
            o_z.z = dist2_10;
            o_z.w = dist2_00;
        }
        cost_x(gx,gy,gz) = weight*inv_spacing2.x*o_x;
        cost_y(gx,gy,gz) = weight*inv_spacing2.y*o_y;
        cost_z(gx,gy,gz) = weight*inv_spacing2.z*o_z;
    }

     // Compute cost at block border
    
     if (x == 0 && gx != 0) {
        float4 dx = df(gx-1, gy, gz) - initial_df(gx-1, gy, gz);
        
         float4 diff_00 = d - dx;
         float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        
         float4 diff_01 = (d+delta4) - dx;
         float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        
         cost_x(gx-1,gy,gz).x = weight*inv_spacing2.x*dist2_00;
         cost_x(gx-1,gy,gz).y = weight*inv_spacing2.x*dist2_01;
         cost_x(gx-1,gy,gz).z = weight*inv_spacing2.x*dist2_00; // border nodes can't move
         cost_x(gx-1,gy,gz).w = cost_x(gx-1,gy,gz).x;
     }
    
     if (y == 0 && gy != 0) {
        float4 dy = df(gx, gy-1, gz) - initial_df(gx, gy-1, gz);
        
         float4 diff_00 = d - dy;
         float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        
         float4 diff_01 = (d+delta4) - dy;
         float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        
         cost_y(gx,gy-1,gz).x = weight*inv_spacing2.y*dist2_00;
         cost_y(gx,gy-1,gz).y = weight*inv_spacing2.y*dist2_01;
         cost_y(gx,gy-1,gz).z = weight*inv_spacing2.y*dist2_00; // border nodes can't move
         cost_y(gx,gy-1,gz).w = cost_x(gx,gy-1,gz).x;
     }

     if (z == 0 && gz != 0) {
        float4 dz = df(gx, gy, gz-1) - initial_df(gx, gy, gz-1);
        
         float4 diff_00 = d - dz;
         float dist2_00 = diff_00.x*diff_00.x + diff_00.y*diff_00.y + diff_00.z*diff_00.z;
        
         float4 diff_01 = (d+delta4) - dz;
         float dist2_01 = diff_01.x*diff_01.x + diff_01.y*diff_01.y + diff_01.z*diff_01.z;
        
         cost_z(gx,gy,gz-1).x = weight*inv_spacing2.z*dist2_00;
         cost_z(gx,gy,gz-1).y = weight*inv_spacing2.z*dist2_01;
         cost_z(gx,gy,gz-1).z = weight*inv_spacing2.z*dist2_00; // border nodes can't move
         cost_z(gx,gy,gz-1).w = cost_x(gx,gy,gz-1).x;
     }
}

template<typename T>
__global__ void ssd_kernel(
    cuda::VolumePtr<T> fixed,
    cuda::VolumePtr<T> moving,
    cuda::VolumePtr<float4> df,
    float3 delta,
    float weight,
    int3 offset,
    int3 dims,
    dim3 moving_dims,
    float3 fixed_origin,
    float3 fixed_spacing,
    float3 moving_origin,
    float3 inv_moving_spacing,
    cuda::VolumePtr<float2> cost_acc
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

    x += offset.x;
    y += offset.y;
    z += offset.z;

    float3 d0 { df(x,y,z).x, df(x,y,z).y, df(x,y,z).z };
    float3 d1 = d0 + delta;

    float3 world_p = fixed_origin + float3{float(x),float(y),float(z)} * fixed_spacing; 
    
    float3 moving_p0 = (world_p + d0 - moving_origin) * inv_moving_spacing; 
    float3 moving_p1 = (world_p + d1 - moving_origin) * inv_moving_spacing; 

    float f0 = fixed(x,y,z) - cuda::linear_at_border<float>(
        moving, moving_dims, moving_p0.x, moving_p0.y, moving_p0.z);
    
    float f1 = fixed(x,y,z) - cuda::linear_at_border<float>(
        moving, moving_dims, moving_p1.x, moving_p1.y, moving_p1.z);
    
    cost_acc(x,y,z).x = weight*f0*f0;
    cost_acc(x,y,z).y = weight*f1*f1;
}

template<typename T>
__global__ void ncc_kernel(
    cuda::VolumePtr<T> fixed,
    cuda::VolumePtr<T> moving,
    cuda::VolumePtr<float4> df,
    float3 delta,
    float weight,
    int radius,
    int3 offset,
    int3 dims,
    dim3 fixed_dims,
    dim3 moving_dims,
    float3 fixed_origin,
    float3 fixed_spacing,
    float3 moving_origin,
    float3 inv_moving_spacing,
    cuda::VolumePtr<float2> cost_acc
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

    x += offset.x;
    y += offset.y;
    z += offset.z;

    float3 d0 { df(x,y,z).x, df(x, y, z).y, df(x, y, z).z };
    float3 d1 = d0 + delta;

    float3 world_p = fixed_origin + float3{float(x),float(y),float(z)} * fixed_spacing; 
    
    float3 moving_p0 = (world_p + d0 - moving_origin) * inv_moving_spacing; 
    float3 moving_p1 = (world_p + d1 - moving_origin) * inv_moving_spacing; 
    
    float sff = 0.0f;
    float sf = 0.0f;
    
    float smm0 = 0.0f;
    float smm1 = 0.0f;
    float sfm0 = 0.0f;
    float sfm1 = 0.0f;
    float sm0 = 0.0f;
    float sm1 = 0.0f;

    unsigned int n = 0;

    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                // TODO: Does not account for anisotropic volumes
                int r2 = dx*dx + dy*dy + dz*dz;
                if (r2 > radius * radius)
                    continue;

                int3 fp{x + dx, y + dy, z + dz};
                
                if (fp.x < 0 || fp.x >= int(fixed_dims.x) ||
                    fp.y < 0 || fp.y >= int(fixed_dims.y) ||
                    fp.z < 0 || fp.z >= int(fixed_dims.z))
                    continue;

                float3 mp0{moving_p0.x + dx, moving_p0.y + dy, moving_p0.z + dz};
                float3 mp1{moving_p1.x + dx, moving_p1.y + dy, moving_p1.z + dz};

                float fixed_v = fixed(fp.x, fp.y, fp.z);

                float moving_v0 = cuda::linear_at_border<float>(moving, moving_dims, mp0.x, mp0.y, mp0.z);
                float moving_v1 = cuda::linear_at_border<float>(moving, moving_dims, mp1.x, mp1.y, mp1.z);

                sff += fixed_v * fixed_v;

                smm0 += moving_v0 * moving_v0;
                smm1 += moving_v1 * moving_v1;
                
                sfm0 += fixed_v*moving_v0;
                sfm1 += fixed_v*moving_v1;

                sm0 += moving_v0;
                sm1 += moving_v1;

                sf += fixed_v;

                ++n;
            }
        }
    }

    if (n == 0)
        return;

    // Subtract mean
    sff -= (sf * sf / n);
    smm0 -= (sm0 * sm0 / n);
    sfm0 -= (sf * sm0 / n);
    smm1 -= (sm1 * sm1 / n);
    sfm1 -= (sf * sm1 / n);
    
    float denom1 = sqrt(sff*smm1);
    float denom0 = sqrt(sff*smm0);

    // Set cost to zero if outside moving volume
    
    if (moving_p0.x >= 0 && moving_p0.x < moving_dims.x &&
        moving_p0.y >= 0 && moving_p0.y < moving_dims.y &&
        moving_p0.z >= 0 && moving_p0.z < moving_dims.z &&
        denom0 > 1e-5)
    {
        cost_acc(x,y,z).x += weight * 0.5f * (1.0f-float(sfm0 / denom0));
    }

    if (moving_p1.x >= 0 && moving_p1.x < moving_dims.x &&
        moving_p1.y >= 0 && moving_p1.y < moving_dims.y &&
        moving_p1.z >= 0 && moving_p1.z < moving_dims.z &&
        denom1 > 1e-5)
    {
        cost_acc(x,y,z).y += weight * 0.5f * (1.0f-float(sfm1 / denom1));
    }
}


void GpuCostFunction_SSD::cost(
    stk::GpuVolume& df,
    const float3& delta,
    float weight, 
    const int3& offset,
    const int3& dims,
    stk::GpuVolume& cost_acc,
    stk::cuda::Stream& stream
)
{
    ASSERT(_fixed.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(_moving.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(df.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(cost_acc.voxel_type() == stk::Type_Float2);

    FATAL_IF(_fixed.voxel_type() != stk::Type_Float || _moving.voxel_type() != stk::Type_Float)
        << "Unsupported format";

    dim3 block_size {32, 32, 1};

    if (dims.x <= 16 || dims.y <= 16) {
        block_size = {16, 16, 4};
    }

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    float3 inv_moving_spacing = {
        1.0f / _moving.spacing().x,
        1.0f / _moving.spacing().y,
        1.0f / _moving.spacing().z
    };

    ssd_kernel<float><<<grid_size, block_size, 0, stream>>>(
        _fixed,
        _moving,
        df,
        delta,
        weight,
        offset,
        dims,
        _moving.size(),
        _fixed.origin(),
        _fixed.spacing(),
        _moving.origin(),
        inv_moving_spacing,
        cost_acc
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

void GpuCostFunction_NCC::cost(
    stk::GpuVolume& df,
    const float3& delta,
    float weight, 
    const int3& offset,
    const int3& dims,
    stk::GpuVolume& cost_acc,
    stk::cuda::Stream& stream
)
{
    ASSERT(_fixed.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(_moving.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(df.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(cost_acc.voxel_type() == stk::Type_Float2);

    FATAL_IF(_fixed.voxel_type() != stk::Type_Float || _moving.voxel_type() != stk::Type_Float)
        << "Unsupported format";

    dim3 block_size {32, 32, 1};

    if (dims.x <= 16 || dims.y <= 16) {
        block_size = {16, 16, 4};
    }

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    float3 inv_moving_spacing = {
        1.0f / _moving.spacing().x,
        1.0f / _moving.spacing().y,
        1.0f / _moving.spacing().z
    };

    ncc_kernel<float><<<grid_size, block_size, 0, stream>>>(
        _fixed,
        _moving,
        df,
        delta,
        weight,
        _radius,
        offset,
        dims,
        _fixed.size(),
        _moving.size(),
        _fixed.origin(),
        _fixed.spacing(),
        _moving.origin(),
        inv_moving_spacing,
        cost_acc
    );
    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

void GpuBinaryFunction::operator()(
    const stk::GpuVolume& df,
    const float3& delta,
    const int3& offset,
    const int3& dims,
    stk::GpuVolume& cost_x,
    stk::GpuVolume& cost_y,
    stk::GpuVolume& cost_z,
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

    float3 inv_spacing2 {
        1.0f / (_spacing.x*_spacing.x),
        1.0f / (_spacing.y*_spacing.y),
        1.0f / (_spacing.z*_spacing.z)
    };

    regularizer_kernel<<<grid_size, block_size, 0, stream>>>(
        df,
        _initial,
        delta,
        _weight,
        offset,
        dims,
        df.size(),
        inv_spacing2,
        cost_x,
        cost_y,
        cost_z
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}
