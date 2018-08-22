#include "resample.h"

#include <stk/common/assert.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/float4.h>

#include "gaussian_filter.h"

#include <algorithm>

#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    #error Displacement residuals not implemented for CUDA
#endif

namespace cuda = stk::cuda;

template<typename T>
__global__ void shrink_volume_by_2_kernel(
    const cuda::VolumePtr<T> in,
    dim3 new_dims,
    cuda::VolumePtr<T> out
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= new_dims.x ||
        y >= new_dims.y ||
        z >= new_dims.z)
    {
        return;
    }

    out(x, y, z) = in(int(2*x), int(2*y), int(2*z));
}

__global__ void upsample_vectorfield_kernel(
    cudaTextureObject_t src,
    dim3 new_dims,
    float3 inv_scale,
    cuda::VolumePtr<float4> out
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= new_dims.x ||
        y >= new_dims.y ||
        z >= new_dims.z)
    {
        return;
    }

    out(x, y, z) = tex3D<float4>(src, x * inv_scale.x + 0.5f, y * inv_scale.y + 0.5f, z * inv_scale.z + 0.5f);
}

namespace {

/// Shrinks the volume by removing every other element
stk::GpuVolume shrink_volume_by_2(const stk::GpuVolume& src)
{
    ASSERT(src.voxel_type() == stk::Type_Float ||
           src.voxel_type() == stk::Type_Float4);
    ASSERT(src.usage() == stk::gpu::Usage_PitchedPointer);

    dim3 old_dims = src.size();
    dim3 new_dims {
        uint32_t(ceil(old_dims.x * 0.5f)),
        uint32_t(ceil(old_dims.y * 0.5f)),
        uint32_t(ceil(old_dims.z * 0.5f)),
    };

    stk::GpuVolume dest(new_dims, src.voxel_type());
    dest.copy_meta_from(src);
    
    float3 old_spacing = src.spacing();
    float3 new_spacing {
        old_spacing.x * (old_dims.x / float(new_dims.x)),
        old_spacing.y * (old_dims.y / float(new_dims.y)),
        old_spacing.z * (old_dims.z / float(new_dims.z))
    };
    dest.set_spacing(new_spacing);

    dim3 block_size{8,8,1};
    dim3 grid_size {
        (new_dims.x + block_size.x - 1) / block_size.x,
        (new_dims.y + block_size.y - 1) / block_size.y,
        (new_dims.z + block_size.z - 1) / block_size.z
    };

    if (src.voxel_type() == stk::Type_Float) {
        shrink_volume_by_2_kernel<float><<<grid_size, block_size>>>(
            src,
            new_dims,
            dest
        );
    }
    else if (src.voxel_type() == stk::Type_Float4) {
        shrink_volume_by_2_kernel<float4><<<grid_size, block_size>>>(
            src,
            new_dims,
            dest
        );
    }
    else {
        ASSERT(false);
    }
    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

    return dest;
}

}


namespace filters {
namespace gpu {
    stk::GpuVolume downsample_volume_by_2(const stk::GpuVolume& vol)
    {
        ASSERT(vol.voxel_type() == stk::Type_Float);
        ASSERT(vol.usage() == stk::gpu::Usage_PitchedPointer);
    
        float3 spacing = vol.spacing();
        float unit_sigma = std::min(spacing.x, std::min(spacing.y, spacing.z));

        stk::GpuVolume filtered = gaussian_filter_3d(vol, unit_sigma);

        return shrink_volume_by_2(filtered);
    }

    stk::GpuVolume downsample_vectorfield_by_2(const stk::GpuVolume& vol
#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    , stk::GpuVolume& residual
#endif
    )
    {
        ASSERT(vol.voxel_type() == stk::Type_Float4);
        ASSERT(vol.usage() == stk::gpu::Usage_PitchedPointer);
    
        float3 spacing = vol.spacing();
        float unit_sigma = std::min(spacing.x, std::min(spacing.y, spacing.z));

        stk::GpuVolume filtered = gaussian_filter_3d(vol, unit_sigma);

        return shrink_volume_by_2(filtered);
    }

    stk::GpuVolume upsample_vectorfield(const stk::GpuVolume& vol, const dim3& new_dims
#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    , const stk::GpuVolume& residual
#endif
    )
    {
        ASSERT(vol.voxel_type() == stk::Type_Float4); // No float3 in gpu volumes
        ASSERT(vol.usage() == stk::gpu::Usage_PitchedPointer);
        
        dim3 old_dims = vol.size();
        float3 inv_scale{
            float(old_dims.x) / new_dims.x,
            float(old_dims.y) / new_dims.y,
            float(old_dims.z) / new_dims.z
        };
        
        stk::GpuVolume out(new_dims, stk::Type_Float4, stk::gpu::Usage_PitchedPointer);
        out.copy_meta_from(vol);

        float3 old_spacing = vol.spacing();
        float3 new_spacing{
            old_spacing.x * inv_scale.x,
            old_spacing.y * inv_scale.y,
            old_spacing.z * inv_scale.z
        };
        out.set_spacing(new_spacing);

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.addressMode[2] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;


        // TODO: This will cause a copy of the input data, performance impact should be investigated
        // Probably hard to avoid unless we write our own interpolation routine inside the kernel.
        cuda::TextureObject src_obj(
            vol.as_usage(stk::gpu::Usage_Texture), 
            tex_desc
        );

        dim3 block_size{8,8,1};
        dim3 grid_size {
            (new_dims.x + block_size.x - 1) / block_size.x,
            (new_dims.y + block_size.y - 1) / block_size.y,
            (new_dims.z + block_size.z - 1) / block_size.z
        };

        upsample_vectorfield_kernel<<<grid_size, block_size>>>(
            src_obj,
            new_dims,
            inv_scale,
            out
        );

        CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

        return out;
    }
}
}