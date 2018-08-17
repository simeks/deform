#include "cost_function.h"

#include <stk/common/assert.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/ptr.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/float4.h>

namespace cuda = stk::cuda;

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
    
    cost_acc(x,y,z) = f*f;
}

void gpu::run_ssd_kernel(
    const stk::GpuVolume& fixed,
    const stk::GpuVolume& moving,
    const stk::GpuVolume& df,
    stk::GpuVolume& cost_acc
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

    dim3 block_size{32,32,1};
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
