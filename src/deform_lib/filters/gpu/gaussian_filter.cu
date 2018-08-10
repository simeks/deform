#include "gaussian_filter.h"

#include <stk/common/assert.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/ptr.h>
#include <stk/image/gpu_volume.h>

namespace cuda = stk::cuda;

__global__ void gaussian_filter_x_kernel(
    dim3 dims,
    const cuda::VolumePtr<float> in,
    int filter_size,
    float factor,
    float spacing,
    cuda::VolumePtr<float> out
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

    float value = 0.0f;
    float norm = 0.0f;
    for (int t = -filter_size; t < filter_size + 1; ++t)
    {
        float c = expf(factor*t*t*spacing*spacing);

        int sx = max(0, min(x + t, dims.x - 1));
        value += c * in(sx, y, z);
        norm += c;
    }
    out(x, y, z) = value / norm;
}
__global__ void gaussian_filter_y_kernel(
    dim3 dims,
    const cuda::VolumePtr<float> in,
    int filter_size,
    float factor,
    float spacing,
    cuda::VolumePtr<float> out
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

    float value = 0.0f;
    float norm = 0.0f;
    for (int t = -filter_size; t < filter_size + 1; ++t)
    {
        float c = expf(factor*t*t*spacing*spacing);

        int sy = max(0, min(y + t, dims.y - 1));
        value += c * in(x, sy, z);
        norm += c;
    }
    out(x, y, z) = value / norm;
}
__global__ void gaussian_filter_z_kernel(
    dim3 dims,
    const cuda::VolumePtr<float> in,
    int filter_size,
    float factor,
    float spacing,
    cuda::VolumePtr<float> out
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

    float value = 0.0f;
    float norm = 0.0f;
    for (int t = -filter_size; t < filter_size + 1; ++t)
    {
        float c = expf(factor*t*t*spacing*spacing);

        int sz = max(0, min(z + t, dims.z - 1));
        value += c * in(x, y, sz);
        norm += c;
    }
    out(x, y, z) = value / norm;
}


namespace filters
{
namespace gpu
{
    stk::GpuVolume gaussian_filter_3d(const stk::GpuVolume& volume, float sigma)
    {
        ASSERT(volume.valid());
        ASSERT(volume.voxel_type() == stk::Type_Float); // Only float32 supported for now
        ASSERT(volume.usage() == stk::gpu::Usage_PitchedPointer);

        stk::GpuVolume tmp = volume.clone_as(stk::gpu::Usage_PitchedPointer);
        stk::GpuVolume out = volume.clone_as(stk::gpu::Usage_PitchedPointer);

        dim3 volume_size = volume.size();
        float3 spacing = volume.spacing();
        
        int3 filter_size{
            (int)ceilf(3 * sigma / spacing.x),
            (int)ceilf(3 * sigma / spacing.y),
            (int)ceilf(3 * sigma / spacing.z)
        };
    
        dim3 block_size{8,8,1};
        dim3 grid_size
        {
            (volume_size.x + block_size.x - 1) / block_size.x,
            (volume_size.y + block_size.y - 1) / block_size.y,
            (volume_size.z + block_size.z - 1) / block_size.z
        };

        float factor = -1.0f / (2.0f * sigma * sigma);
        gaussian_filter_x_kernel<<<grid_size, block_size>>>(
            volume_size,
            tmp.pitched_ptr(),
            filter_size.x,
            factor,
            spacing.x,
            out.pitched_ptr()
        );
        
        CUDA_CHECK_ERRORS(cudaDeviceSynchronize());
        tmp.copy_from(out);

        gaussian_filter_y_kernel<<<grid_size, block_size>>>(
            volume_size,
            tmp.pitched_ptr(),
            filter_size.y,
            factor,
            spacing.y,
            out.pitched_ptr()
        );

        CUDA_CHECK_ERRORS(cudaDeviceSynchronize());
        tmp.copy_from(out);

        gaussian_filter_z_kernel<<<grid_size, block_size>>>(
            volume_size,
            tmp.pitched_ptr(),
            filter_size.z,
            factor,
            spacing.z,
            out.pitched_ptr()
        );

        CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

        out = out.as_usage(volume.usage());

        return out;
    }
}
}