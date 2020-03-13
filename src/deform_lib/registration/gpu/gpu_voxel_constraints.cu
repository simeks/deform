#include "../gpu_volume_pyramid.h"
#include "gpu_displacement_field.h"
#include "gpu_voxel_constraints.h"

#include <stk/cuda/cuda.h>
#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>

__global__ void constrain_displacement_field(
    const stk::cuda::VolumePtr<uint8_t> mask,
    const stk::cuda::VolumePtr<float4> values,
    dim3 dims,
    stk::cuda::VolumePtr<float4> df
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= dims.x ||
        y >= dims.y ||
        z >= dims.z) {
        return;
    }

    if (mask(x,y,z) == 1)
        df(x,y,z) = values(x,y,z);
}

__global__ void downsample_mask_by_2_kernel(
    const stk::cuda::VolumePtr<uint8_t> mask,
    dim3 old_dims,
    dim3 new_dims,
    stk::cuda::VolumePtr<uint8_t> out
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= new_dims.x ||
        y >= new_dims.y ||
        z >= new_dims.z) {
        return;
    }

    int3 subvoxels[] = {
        int3{0, 0, 0},
        int3{0, 0, 1},
        int3{0, 1, 0},
        int3{0, 1, 1},
        int3{1, 0, 0},
        int3{1, 0, 1},
        int3{1, 1, 0},
        int3{1, 1, 1}
    };

    int3 src_p{2*x, 2*y, 2*z};

    uint8_t m = 0;
    for (int i = 0; i < 8; ++i) {
        int3 p = src_p + subvoxels[i];
        if (p.x >= int(old_dims.x) ||
            p.y >= int(old_dims.y) ||
            p.z >= int(old_dims.z))
            continue;

        m = max(m, mask(p.x, p.y, p.z));
    }
    out(x, y, z) = m;
}

__global__ void downsample_values_by_2_kernel(
    const stk::cuda::VolumePtr<uint8_t> mask,
    const stk::cuda::VolumePtr<float4> values,
    dim3 old_dims,
    dim3 new_dims,
    stk::cuda::VolumePtr<float4> result
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= new_dims.x ||
        y >= new_dims.y ||
        z >= new_dims.z) {
        return;
    }

    const int3 subvoxels[] = {
        int3{0, 0, 0},
        int3{0, 0, 1},
        int3{0, 1, 0},
        int3{0, 1, 1},
        int3{1, 0, 0},
        int3{1, 0, 1},
        int3{1, 1, 0},
        int3{1, 1, 1}
    };

    int3 src_p{2*x, 2*y, 2*z};

    int nmask = 0;
    float4 val{0,0,0,0};
    for (int i = 0; i < 8; ++i) {
        int3 p = src_p + subvoxels[i];
        if (p.x >= int(old_dims.x) ||
            p.y >= int(old_dims.y) ||
            p.z >= int(old_dims.z))
            continue;

        if (mask(p.x, p.y, p.z) > 0) {
            ++nmask;
            val = val + values(p.x, p.y, p.z);
        }
    }
    if (nmask > 0) {
        result(x, y, z) = val / float(nmask);
    } else {
        result(x, y, z) = float4{0,0,0,0};
    }

}



namespace {
stk::GpuVolume downsample_mask_by_2(const stk::GpuVolume& mask)
{
    FATAL_IF(mask.voxel_type() != stk::Type_UChar)
        << "Invalid voxel type";
    /*
        Downsampling of mask volumes by a factor of 2. Designed specifically for binary masks.
        Each resulting downsampled voxel takes the max value from the corresponding subvoxels
    */

    dim3 old_dims = mask.size();
    dim3 new_dims{
        uint32_t(ceil(old_dims.x * 0.5f)),
        uint32_t(ceil(old_dims.y * 0.5f)),
        uint32_t(ceil(old_dims.z * 0.5f)),
    };

    stk::GpuVolume result(new_dims, mask.voxel_type());
    result.copy_meta_from(mask);

    float3 old_spacing = mask.spacing();
    float3 new_spacing {
        old_spacing.x * (old_dims.x / float(new_dims.x)),
        old_spacing.y * (old_dims.y / float(new_dims.y)),
        old_spacing.z * (old_dims.z / float(new_dims.z))
    };
    result.set_spacing(new_spacing);

    dim3 block_size{32,32,1};
    dim3 grid_size {
        (new_dims.x + block_size.x - 1) / block_size.x,
        (new_dims.y + block_size.y - 1) / block_size.y,
        (new_dims.z + block_size.z - 1) / block_size.z
    };
    
    downsample_mask_by_2_kernel<<<grid_size, block_size>>>(
        mask,
        old_dims,
        new_dims,
        result
    );

    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

    return result;
}
stk::GpuVolume downsample_values_by_2(
    const stk::GpuVolume& mask, const stk::GpuVolume& values)
{
    /*
        Downsamples a constraint vector field.
        The value of each downsampled voxel is calculated as the mean of all subvoxels
            that are flagged as constraints (1s in the mask).
    */

    dim3 old_dims = mask.size();
    dim3 new_dims{
        uint32_t(ceil(old_dims.x * 0.5f)),
        uint32_t(ceil(old_dims.y * 0.5f)),
        uint32_t(ceil(old_dims.z * 0.5f)),
    };

    stk::GpuVolume result(new_dims, values.voxel_type());
    result.copy_meta_from(mask);

    float3 old_spacing = values.spacing();
    float3 new_spacing {
        old_spacing.x * (old_dims.x / float(new_dims.x)),
        old_spacing.y * (old_dims.y / float(new_dims.y)),
        old_spacing.z * (old_dims.z / float(new_dims.z))
    };
    result.set_spacing(new_spacing);

    dim3 block_size{32,32,1};
    dim3 grid_size {
        (new_dims.x + block_size.x - 1) / block_size.x,
        (new_dims.y + block_size.y - 1) / block_size.y,
        (new_dims.z + block_size.z - 1) / block_size.z
    };

    downsample_values_by_2_kernel<<<grid_size, block_size>>>(
        mask,
        values,
        old_dims,
        new_dims,
        result
    );

    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

    return result;
}
}

void gpu_voxel_constraints::constraint_displacement_field(
    GpuDisplacementField& df,
    const stk::GpuVolume& mask,
    const stk::GpuVolume& values)
{
    if (mask.voxel_type() != stk::Type_UChar 
    || values.voxel_type() != stk::Type_Float4
    || mask.size() != values.size()) {
        FATAL() << "Invalid constraint volumes";
    }
    if (df.size() != values.size()) {
        FATAL() << "Invalid displacement field volume";
    }

    dim3 dims = df.size();

    dim3 block_size{32,32,1};
    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    constrain_displacement_field<<<grid_size, block_size>>>(
        mask,
        values,
        dims,
        df.volume()
    );
    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());
}

void gpu_voxel_constraints::build_pyramids(
    const stk::GpuVolume& mask,
    const stk::GpuVolume& values,
    int num_levels,
    GpuVolumePyramid& mask_pyramid,
    GpuVolumePyramid& values_pyramid)
{
    if (mask.voxel_type() != stk::Type_UChar 
    || values.voxel_type() != stk::Type_Float4
    || mask.size() != values.size()) {
        FATAL() << "Invalid constraint volumes";
    }

    mask_pyramid.set_level_count(num_levels);
    values_pyramid.set_level_count(num_levels);

    mask_pyramid.set_volume(0, mask);
    values_pyramid.set_volume(0, values);

    for (int i = 0; i < num_levels-1; ++i) {
        stk::GpuVolume prev_mask = mask_pyramid.volume(i);
        stk::GpuVolume prev_values = values_pyramid.volume(i);

        mask_pyramid.set_volume(i+1, downsample_mask_by_2(prev_mask));
        values_pyramid.set_volume(i+1, downsample_values_by_2(prev_mask, prev_values));
    }
}
