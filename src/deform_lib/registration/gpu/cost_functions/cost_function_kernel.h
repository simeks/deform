#pragma once

#include <deform_lib/registration/settings.h>

#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/matrix3x3f.h>

#include <cfloat>

#include "../gpu_displacement_field.h"

namespace cuda {
    using namespace stk::cuda;
}

// Helper class for implementing intensity-based cost functions on CUDA
template<typename TImpl>
struct CostFunctionKernel
{
    using VoxelType = typename TImpl::VoxelType;

    CostFunctionKernel(
        const TImpl& impl,
        const stk::GpuVolume& fixed,
        const stk::GpuVolume& moving,
        const stk::GpuVolume& fixed_mask,
        const stk::GpuVolume& moving_mask,
        float weight,
        stk::GpuVolume& cost // float2
    ) :
        _impl(impl),
        _fixed(fixed),
        _moving(moving),
        _fixed_dims(fixed.size()),
        _moving_dims(moving.size()),
        _fixed_mask(fixed_mask),
        _moving_mask(moving_mask),
        _fixed_origin(fixed.origin()),
        _fixed_spacing(fixed.spacing()),
        _fixed_direction(fixed.direction()),
        _moving_origin(moving.origin()),
        _inv_moving_spacing(float3{1.0f, 1.0f, 1.0f} / moving.spacing()),
        _inv_moving_direction(moving.inverse_direction()),
        _weight(weight),
        _cost(cost)
    {
        ASSERT(cost.voxel_type() == stk::Type_Float2);
    }
    ~CostFunctionKernel()
    {
    }

    // x,y,z        : Global image coordinates
    // d            : Displacement
    // cost_offset  : Where in `cost` to store the results (0 or 1).
    template<typename TDisplacementField>
    __device__ void operator()(
        int x,
        int y,
        int z,
        TDisplacementField df,
        int cost_offset)
    {

        // Check if the fixed voxel is masked out
        float fixed_mask_value = 1.0f;
        if (_fixed_mask.ptr) {
            fixed_mask_value = _fixed_mask(x, y, z);
            if (fixed_mask_value <= FLT_EPSILON) {
                return;
            }
        }

        const float3 moving_p = _inv_moving_direction
            * (df.transform_index({x,y,z}) - _moving_origin)
            * _inv_moving_spacing;

        // Check if the moving voxels are masked out
        float moving_mask_value = 1.0f;
        if (_moving_mask.ptr) {
            moving_mask_value = cuda::linear_at_border<float>(
                    _moving_mask, _moving_dims, moving_p.x, moving_p.y, moving_p.z);

            if (moving_mask_value <= FLT_EPSILON) {
                return;
            }
        }

        float3 d = df.get(int3{x, y, z}, delta);
        float c = _impl(
            _fixed,
            _moving,
            _fixed_dims,
            _moving_dims,
            int3{x,y,z},
            moving_p,
            d
        );
        c *= _weight * fixed_mask_value * moving_mask_value;

        reinterpret_cast<float*>(&_cost(x,y,z))[cost_offset] += c;
    }

    TImpl _impl;

    cuda::VolumePtr<VoxelType> _fixed;
    cuda::VolumePtr<VoxelType> _moving;
    dim3 _fixed_dims;
    dim3 _moving_dims;

    cuda::VolumePtr<float> _fixed_mask;
    cuda::VolumePtr<float> _moving_mask;

    float3 _moving_origin;
    float3 _inv_moving_spacing;
    Matrix3x3f _inv_moving_direction;

    float _weight;

    cuda::VolumePtr<float2> _cost;
};

template<typename TKernel, typename TDisplacementField>
__global__ void cost_function_kernel(
    TKernel kernel,
    int3 offset,
    int3 dims,
    TDisplacementField df,
    float4 delta,
    int cost_offset)
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

    float3 inv_spacing {
        1.0f / kernel._fixed_spacing.x,
        1.0f / kernel._fixed_spacing.y,
        1.0f / kernel._fixed_spacing.z
    };

    kernel(x, y, z, df, cost_offset);
}

template<typename TKernel>
void invoke_cost_function_kernel(
    const TKernel& kernel,
    const float3& delta,
    const int3& offset,
    const int3& dims,
    GpuDisplacementField& df,
    stk::cuda::Stream& stream
)
{
    dim3 block_size {32, 32, 1};

    if (dims.x <= 16 || dims.y <= 16) {
        block_size = {16, 16, 4};
    }

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    // Same for both compositive and additive
    cost_function_kernel
    <<<grid_size, block_size, 0, stream>>>(
        kernel,
        offset,
        dims,
        cuda::DisplacementField<>(df),
        float4{0,0,0,0},
        0
    );

    float4 d4 { delta.x, delta.y, delta.z, 0 };

    if (df.update_rule() == Settings::UpdateRule_Compositive) {
        cost_function_kernel<<<grid_size, block_size, 0, stream>>>(
            kernel,
            offset,
            dims,
            cuda::DisplacementField<CompositiveUpdate>(df),
            d4,
            1
        );
    }
    else if (df.update_rule() == Settings::UpdateRule_Additive) {
        cost_function_kernel<<<grid_size, block_size, 0, stream>>>(
            kernel,
            offset,
            dims,
            cuda::DisplacementField<AdditiveUpdate>(df),
            d4,
            1
        );
    }
    else {
        ASSERT(false);
    }

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}
