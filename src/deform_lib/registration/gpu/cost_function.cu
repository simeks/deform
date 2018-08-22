#include "cost_function.h"

#include <stk/common/assert.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/float4.h>

namespace cuda = stk::cuda;

__global__ void regularizer_kernel(
    cuda::VolumePtr<float4> df,
    cuda::VolumePtr<float4> initial_df,
    dim3 dims,
    float3 spacing,
    cuda::VolumePtr<float4> out // Regularization cost in x+,y+,z+
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

    float4 o = {0, 0, 0, 0};

    if (x + 1 < dims.x) {
        float4 diff_x = (df(x,y,z) - initial_df(x,y,z)) - 
                        (df(x+1,y,z) - initial_df(x+1,y,z));
        float dist2_x = diff_x.x*diff_x.x + diff_x.y*diff_x.y + diff_x.z*diff_x.z;
        o.x = dist2_x / (spacing.x*spacing.x);
    }
    if (y + 1 < dims.y) {
        float4 diff_y = (df(x,y,z) - initial_df(x,y,z)) - 
                        (df(x,y+1,z) - initial_df(x,y+1,z));
        float dist2_y = diff_y.x*diff_y.x + diff_y.y*diff_y.y + diff_y.z*diff_y.z;
        o.y = dist2_y / (spacing.y*spacing.y);
    }
    if (z + 1 < dims.z) {
        float4 diff_z = (df(x,y,z) - initial_df(x,y,z)) - 
                        (df(x,y,z+1) - initial_df(x,y,z+1));
        float dist2_z = diff_z.x*diff_z.x + diff_z.y*diff_z.y + diff_z.z*diff_z.z;
        o.z = dist2_z / (spacing.z*spacing.z);
    }
    
    o.w = 0;

    out(x,y,z) = o;
}

template<typename T>
__global__ void ssd_kernel(
    cudaTextureObject_t fixed,
    cudaTextureObject_t moving,
    cuda::VolumePtr<float4> df,
    dim3 fixed_dims,
    dim3 moving_dims,
    float4 fixed_origin,
    float4 fixed_spacing,
    float4 moving_origin,
    float4 moving_spacing,
    cuda::VolumePtr<float> cost_acc
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= fixed_dims.x ||
        y >= fixed_dims.y ||
        z >= fixed_dims.z)
    {
        return;
    }

    float4 world_p = fixed_origin + float4{float(x),float(y),float(z),0} * fixed_spacing; 
    float4 moving_p = (world_p + df(x,y,z) - moving_origin) / moving_spacing; 
    
    // [Filip]: Addition for partial-body registrations
    if (moving_p.x < 0 || moving_p.x > moving_dims.x || 
        moving_p.y < 0 || moving_p.y > moving_dims.y || 
        moving_p.z < 0 || moving_p.z > moving_dims.z) {
        // Does not affect the cost accumulator
        return;
    }

    float f = tex3D<T>(fixed, x + 0.5f, y + 0.5f, z + 0.5f) 
            - tex3D<T>(moving, moving_p.x + 0.5f, moving_p.y + 0.5f, moving_p.z + 0.5f);
    
    cost_acc(x,y,z) = cost_acc(x,y,z) + f*f;
}

template<typename T>
__global__ void ncc_kernel(
    cudaTextureObject_t fixed,
    cudaTextureObject_t moving,
    cuda::VolumePtr<float4> df,
    int radius,
    dim3 fixed_dims,
    dim3 moving_dims,
    float4 fixed_origin,
    float4 fixed_spacing,
    float4 moving_origin,
    float4 moving_spacing,
    cuda::VolumePtr<float> cost_acc
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= fixed_dims.x ||
        y >= fixed_dims.y ||
        z >= fixed_dims.z)
    {
        return;
    }

    float4 world_p = fixed_origin + float4{float(x),float(y),float(z),0} * fixed_spacing; 
    float4 moving_p = (world_p + df(x,y,z) - moving_origin) / moving_spacing; 
    
    // [Filip]: Addition for partial-body registrations
    if (moving_p.x < 0 || moving_p.x > moving_dims.x || 
        moving_p.y < 0 || moving_p.y > moving_dims.y || 
        moving_p.z < 0 || moving_p.z > moving_dims.z) {
        // Does not affect the cost accumulator
        return;
    }

    float sff = 0.0;
    float smm = 0.0;
    float sfm = 0.0;
    float sf = 0.0;
    float sm = 0.0;
    unsigned int n = 0;

    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                // TODO: Does not account for anisotropic volumes
                int r2 = dx*dx + dy*dy + dz*dz;
                if (r2 > radius * radius)
                    continue;

                float3 fp{float(x + dx), float(y + dy), float(z + dz)};
                
                // if (!stk::is_inside(_fixed.size(), fp))
                //     continue;

                float3 mp{moving_p.x + dx, moving_p.y + dy, moving_p.z + dz};

                T fixed_v = tex3D<T>(fixed, fp.x+0.5f, fp.y+0.5f, fp.z+0.5f);
                T moving_v = tex3D<T>(moving, mp.x+0.5f, mp.y+0.5f, mp.z+0.5f);

                sff += fixed_v * fixed_v;
                smm += moving_v * moving_v;
                sfm += fixed_v*moving_v;
                sm += moving_v;
                sf += fixed_v;

                ++n;
            }
        }
    }

    if (n == 0)
        return;

    // Subtract mean
    sff -= (sf * sf / n);
    smm -= (sm * sm / n);
    sfm -= (sf * sm / n);
    
    float d = sqrt(sff*smm);

    if(d > 1e-14) {
        cost_acc(x,y,z) = cost_acc(x,y,z) + 0.5f*(1.0f-float(sfm / d));
    }
}


void gpu::run_regularizer_kernel(
    const stk::GpuVolume& df,
    const stk::GpuVolume& initial_df,
    stk::GpuVolume& cost,
    const dim3& block_size
)
{
    ASSERT(df.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(initial_df.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(cost.usage() == stk::gpu::Usage_PitchedPointer);

    FATAL_IF(df.voxel_type() != stk::Type_Float4 || 
             initial_df.voxel_type() != stk::Type_Float4 ||
             cost.voxel_type() != stk::Type_Float4)
        << "Unsupported format";

    dim3 dims = cost.size();

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    regularizer_kernel<<<grid_size, block_size>>>(
        df,
        initial_df,
        dims,
        df.spacing(),
        cost
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());
}

void gpu::run_ssd_kernel(
    const stk::GpuVolume& fixed,
    const stk::GpuVolume& moving,
    const stk::GpuVolume& df,
    stk::GpuVolume& cost_acc,
    const dim3& block_size
)
{
    ASSERT(fixed.usage() == stk::gpu::Usage_Texture);
    ASSERT(moving.usage() == stk::gpu::Usage_Texture);
    ASSERT(df.usage() == stk::gpu::Usage_PitchedPointer);

    FATAL_IF(fixed.voxel_type() != stk::Type_Float || moving.voxel_type() != stk::Type_Float)
        << "Unsupported format";

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeBorder;
    tex_desc.addressMode[1] = cudaAddressModeBorder;
    tex_desc.addressMode[2] = cudaAddressModeBorder;
    tex_desc.filterMode = cudaFilterModeLinear;

    dim3 dims = cost_acc.size();

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    ssd_kernel<float><<<grid_size, block_size>>>(
        cuda::TextureObject(fixed, tex_desc),
        cuda::TextureObject(moving, tex_desc),
        df,
        dims,
        moving.size(),
        float4{fixed.origin().x, fixed.origin().y, fixed.origin().z, 0},
        float4{fixed.spacing().x, fixed.spacing().y, fixed.spacing().z, 1},
        float4{moving.origin().x, moving.origin().y, moving.origin().z, 0},
        float4{moving.spacing().x, moving.spacing().y, moving.spacing().z, 1},
        cost_acc
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());
}

void gpu::run_ncc_kernel(
    const stk::GpuVolume& fixed,
    const stk::GpuVolume& moving,
    const stk::GpuVolume& df,
    int radius,
    stk::GpuVolume& cost_acc,
    const dim3& block_size
)
{
    ASSERT(fixed.usage() == stk::gpu::Usage_Texture);
    ASSERT(moving.usage() == stk::gpu::Usage_Texture);
    ASSERT(df.usage() == stk::gpu::Usage_PitchedPointer);

    FATAL_IF(fixed.voxel_type() != stk::Type_Float || moving.voxel_type() != stk::Type_Float)
        << "Unsupported format";

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeBorder;
    tex_desc.addressMode[1] = cudaAddressModeBorder;
    tex_desc.addressMode[2] = cudaAddressModeBorder;
    tex_desc.filterMode = cudaFilterModeLinear;

    dim3 dims = cost_acc.size();

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    ncc_kernel<float><<<grid_size, block_size>>>(
        cuda::TextureObject(fixed, tex_desc),
        cuda::TextureObject(moving, tex_desc),
        df,
        radius,
        dims,
        moving.size(),
        float4{fixed.origin().x, fixed.origin().y, fixed.origin().z, 0},
        float4{fixed.spacing().x, fixed.spacing().y, fixed.spacing().z, 1},
        float4{moving.origin().x, moving.origin().y, moving.origin().z, 0},
        float4{moving.spacing().x, moving.spacing().y, moving.spacing().z, 1},
        cost_acc
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());
}
