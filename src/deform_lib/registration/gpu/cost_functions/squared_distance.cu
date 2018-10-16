#include "squared_distance.h"

#include <float.h>

namespace cuda = stk::cuda;

template<typename T, bool use_fixed_mask, bool use_moving_mask>
__global__ void ssd_kernel(
    cuda::VolumePtr<T> fixed,
    cuda::VolumePtr<T> moving,
    cuda::VolumePtr<float> fixed_mask,
    cuda::VolumePtr<float> moving_mask,
    cuda::VolumePtr<float4> df,
    float3 delta,
    float weight,
    int3 offset,
    int3 dims,
    dim3 moving_dims,
    float3 fixed_origin,
    float3 fixed_spacing,
    Matrix3x3f fixed_direction,
    float3 moving_origin,
    float3 inv_moving_spacing,
    Matrix3x3f inv_moving_direction,
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

    // Check if the fixed voxel is masked out
    if (use_fixed_mask) {
        if (fixed_mask(x, y, z) <= FLT_EPSILON) {
            return;
        }
    }

    x += offset.x;
    y += offset.y;
    z += offset.z;

    const float3 d0 { df(x,y,z).x, df(x,y,z).y, df(x,y,z).z };
    const float3 d1 = d0 + delta;

    const float3 xyz = float3{float(x),float(y),float(z)};
    const float3 world_p = fixed_origin + fixed_direction * (xyz * fixed_spacing);

    const float3 moving_p0 = (inv_moving_direction * (world_p + d0 - moving_origin)) * inv_moving_spacing;
    const float3 moving_p1 = (inv_moving_direction * (world_p + d1 - moving_origin)) * inv_moving_spacing;

    // Check if the moving voxels are masked out
    float mask_value_0 = 1.0f;
    float mask_value_1 = 1.0f;
    if (use_moving_mask) {
        mask_value_0 = cuda::linear_at_border<float>(
                moving_mask, moving_dims, moving_p0.x, moving_p0.y, moving_p0.z);
        mask_value_1 = cuda::linear_at_border<float>(
                moving_mask, moving_dims, moving_p1.x, moving_p1.y, moving_p1.z);
    }

    if (mask_value_0 >= FLT_EPSILON) {
        const float f0 = fixed(x,y,z) - cuda::linear_at_border<float>(
            moving, moving_dims, moving_p0.x, moving_p0.y, moving_p0.z);
        cost_acc(x,y,z).x += mask_value_0 * weight * f0*f0;
    }

    if (mask_value_1 >= FLT_EPSILON) {
        const float f1 = fixed(x,y,z) - cuda::linear_at_border<float>(
            moving, moving_dims, moving_p1.x, moving_p1.y, moving_p1.z);
        cost_acc(x,y,z).y += mask_value_1 * weight * f1*f1;
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
    ASSERT(!_fixed_mask.valid() || _fixed_mask.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(!_moving_mask.valid() || _moving_mask.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(cost_acc.voxel_type() == stk::Type_Float2);

    FATAL_IF(_fixed.voxel_type() != stk::Type_Float ||
             _moving.voxel_type() != stk::Type_Float ||
             _fixed_mask.valid() && _fixed_mask.voxel_type() != stk::Type_Float ||
             _moving_mask.valid() && _moving_mask.voxel_type() != stk::Type_Float)
        << "Unsupported pixel type";

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

    auto (*kernel) = &ssd_kernel<float, false, false>;
    if (_fixed_mask.valid()) {
        if (_moving_mask.valid()) {
            kernel = &ssd_kernel<float, true, true>;
        }
        else {
            kernel = &ssd_kernel<float, true, false>;
        }
    }
    else {
        if (_moving_mask.valid()) {
            kernel = &ssd_kernel<float, false, true>;
        }
        else {
            kernel = &ssd_kernel<float, false, false>;
        }
    }

    kernel<<<grid_size, block_size, 0, stream>>>(
        _fixed,
        _moving,
        _fixed_mask.valid() ? _fixed_mask : cuda::VolumePtr<float>::null_ptr(),
        _moving_mask.valid() ? _moving_mask : cuda::VolumePtr<float>::null_ptr(),
        df,
        delta,
        weight,
        offset,
        dims,
        _moving.size(),
        _fixed.origin(),
        _fixed.spacing(),
        _fixed.direction(),
        _moving.origin(),
        inv_moving_spacing,
        _moving.inverse_direction(),
        cost_acc
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

