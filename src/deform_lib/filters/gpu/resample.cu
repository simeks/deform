#include "resample.h"

#include <stk/common/assert.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/ptr.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/float4.h>

#include "gaussian_filter.h"

#include <algorithm>

#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    #error Displacement residuals not implemented for CUDA
#endif

namespace cuda = stk::cuda;

__global__ void downsample_volume_kernel(
    const cuda::VolumePtr<float> in,
    dim3 new_dims,
    float inv_scale,
    cuda::VolumePtr<float> out
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

    out(x, y, z) = in(int(x*inv_scale), int(y*inv_scale), int(z*inv_scale));
}

__global__ void downsample_vectorfield_kernel(
    const cuda::VolumePtr<float4> field,
    dim3 old_dims,
    dim3 new_dims,
    float inv_scale,
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

    int px = int(x * inv_scale);
    int py = int(y * inv_scale);
    int pz = int(z * inv_scale);

    float4 v = field(px, py, pz);

    int px1 = min(px+1, old_dims.x-1);
    int py1 = min(py+1, old_dims.y-1);
    int pz1 = min(pz+1, old_dims.z-1);

    v = v + field(px1, py, pz);
    v = v + field(px, py1, pz);
    v = v + field(px, py, pz1);
    v = v + field(px1, py1, pz);
    v = v + field(px1, py, pz1);
    v = v + field(px, py1, pz1);
    v = v + field(px1, py1, pz1);
    
    float s = 1.0f / 8.0f;
    out(x, y, z) = float4{s*v.x, s*v.y, s*v.z, 0.0f};
}



__global__ void upsample_vectorfield_kernel(
    cudaTextureObject_t src,
    dim3 size,
    float4 scale,
    float4* out
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= size.x ||
        y >= size.y ||
        z >= size.z)
    {
        return;
    }

    //out(x, y, z) = scale * tex3D(src, x / scale.x, y / scale.y, z / scale.z);
}

#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
__global__ void upsample_vectorfield_w_residual_kernel(
    cudaTextureObject_t src,
    dim3 size,
    float4 scale,
    float4* residual,
    float4* out
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= size.x ||
        y >= size.y ||
        z >= size.z)
    {
        return;
    }

    out[x + y * size.width + z * size.width * size.height] 
        = scale * tex3D(_def_texture, x / scale.x, y / scale.y, z / scale.z)
        + residual[x + y * size.width + z * size.width * size.height];
}
#endif

namespace filters {
namespace gpu {
    stk::GpuVolume downsample_volume(const stk::GpuVolume& src, float scale)
    {
        ASSERT(src.voxel_type() == stk::Type_Float);
        ASSERT(scale > 0.0f && scale < 1.0f);
        float inv_scale = 1.0f / scale;
        
        dim3 old_dims = src.size();
        dim3 new_dims {
            uint32_t(ceil(old_dims.x * scale)),
            uint32_t(ceil(old_dims.y * scale)),
            uint32_t(ceil(old_dims.z * scale)),
        };

        stk::GpuVolume dest(new_dims, stk::Type_Float);
        dest.set_origin(src.origin());
        
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

        downsample_volume_kernel<<<grid_size, block_size>>>(
            src.pitched_ptr(),
            new_dims,
            inv_scale,
            dest.pitched_ptr()
        );
        CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

        return dest;
    }

    stk::GpuVolume downsample_volume_gaussian(const stk::GpuVolume& vol, float scale)
    {
        ASSERT(scale > 0.0f && scale < 1.0f);
        ASSERT(vol.voxel_type() == stk::Type_Float);
    
        float3 spacing = vol.spacing();
        float unit_sigma = std::min(spacing.x, std::min(spacing.y, spacing.z));

        stk::GpuVolume src = vol.as_usage(stk::gpu::Usage_PitchedPointer);
        stk::GpuVolume filtered = gaussian_filter_3d(src, unit_sigma);

        return downsample_volume(filtered, scale).as_usage(vol.usage());
    }

    stk::GpuVolume downsample_vectorfield(const stk::GpuVolume& vol, float scale
#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    , stk::GpuVolume& residual
#endif
    )
    {
        ASSERT(scale > 0.0f && scale < 1.0f);
        ASSERT(vol.voxel_type() == stk::Type_Float4); // No float3 in gpu volumes

        float inv_scale = 1.0f / scale;

        dim3 old_dims = vol.size();
        dim3 new_dims {
            uint32_t(ceil(old_dims.x * scale)),
            uint32_t(ceil(old_dims.y * scale)),
            uint32_t(ceil(old_dims.z * scale))
        };

        stk::GpuVolume result(new_dims, stk::Type_Float4);
        result.set_origin(vol.origin());

        float3 old_spacing = vol.spacing();
        float3 new_spacing {
            old_spacing.x * (old_dims.x / float(new_dims.x)),
            old_spacing.y * (old_dims.y / float(new_dims.y)),
            old_spacing.z * (old_dims.z / float(new_dims.z))
        };
        result.set_spacing(new_spacing);

        dim3 block_size{8,8,1};
        dim3 grid_size {
            (new_dims.x + block_size.x - 1) / block_size.x,
            (new_dims.y + block_size.y - 1) / block_size.y,
            (new_dims.z + block_size.z - 1) / block_size.z
        };

        downsample_vectorfield_kernel<<<grid_size, block_size>>>(
            vol.pitched_ptr(),
            old_dims,
            new_dims,
            inv_scale,
            result.pitched_ptr()
        );
        CUDA_CHECK_ERRORS(cudaDeviceSynchronize());
        
        return result;
    }

    stk::GpuVolume upsample_vectorfield(const stk::GpuVolume& vol, const dim3& new_dims
#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    , const stk::GpuVolume& residual
#endif
    )
    {
        vol;new_dims;
        // ASSERT(vol.voxel_type() == voxel::Type_Float4); // No float3 in gpu volumes
        
        // Dims old_dims = vol.size();
        // float4 scale{
        //     new_dims.width / float(old_dims.width),
        //     new_dims.height / float(old_dims.height),
        //     new_dims.depth / float(old_dims.depth),
        //     1.0f
        // };
        
        // // TODO: Any extra cost of BindAsSurface (for returned volume)
        // GpuVolume out(vol.size(), vol.voxel_type());
        // out.set_origin(vol.origin());

        // float3 old_spacing = vol.spacing();
        // float3 new_spacing{
        //     old_spacing.x / scale.x,
        //     old_spacing.y / scale.y,
        //     old_spacing.z / scale.z
        // };
        // out.set_spacing(new_spacing);

        // cudaResourceDesc res_desc{0};
        // res_desc.resType = cudaResourceType;
        // res_desc.array = vol.ptr;
        
        // cudaTextureDesc tex_desc{0};
        // tex_desc.addressMode[0] = cudaAddressModeClamp;
        // tex_desc.addressMode[1] = cudaAddressModeClamp;
        // tex_desc.addressMode[2] = cudaAddressModeClamp;
        // tex_desc.filterMode = cudaFilterModeLinear;

        // cudaTextureObject_t src_obj{0};
        // cudaCreateTextureObject(&src_obj, &res_desc, &tex_desc, nullptr);

        // dim3 block_size{8,8,1};
        // dim3 grid_size
        // {
        //     (new_dims.width + block_size.x - 1) / block_size.x,
        //     (new_dims.height + block_size.y - 1) / block_size.y,
        //     (new_dims.depth + block_size.z - 1) / block_size.z
        // };

        // if (residual.valid())
        // {
        //     assert(false);
        // }
        // else
        // {
        //     upsample_vectorfield_kernel<<<grid_size, block_size>>>(
        //         src_obj,

        //     );
        // }
        return stk::GpuVolume();
    }
}
}