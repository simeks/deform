#pragma once

#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>

namespace cuda = stk::cuda;

// Helper class for implementing intensity-based cost functions on CUDA
template<typename TImpl>
struct CostFunctionKernel
{
    CostFunctionKernel(
        const TImpl& impl,
        const stk::GpuVolume& fixed,
        const stk::GpuVolume& moving,
        const stk::GpuVolume& fixed_mask,
        const stk::GpuVolume& moving_mask,
        const stk::GpuVolume& df,
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
        _df(df),
        _fixed_origin(fixed.origin()),
        _fixed_spacing(fixed.spacing()),
        _fixed_direction(fixed.direction()),
        _moving_origin(moving.origin()),
        _inv_moving_spacing({1.0f / moving.spacing().x, 1.0f / moving.spacing().y, 1.0f / moving.spacing().z}),
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
    // cost_offset  :  Where in `cost` to store the results (0 or 1).
    __device__ void operator()(int x, int y, int z, float3 delta, int cost_offset)
    {
        // Check if the fixed voxel is masked out
        if (_fixed_mask.ptr) {
            if (_fixed_mask(x, y, z) <= FLT_EPSILON) {
                return;
            }
        }

        const float3 d0 {
            _df(x, y, z).x + delta.x,
            _df(x, y, z).y + delta.y,
            _df(x, y, z).z + delta.z
        };

        const float3 xyz = float3{float(x),float(y),float(z)};
        const float3 world_p = _fixed_origin + _fixed_direction * (xyz * _fixed_spacing);
        const float3 moving_p = (_inv_moving_direction * (world_p + d0 - _moving_origin)) 
            * _inv_moving_spacing;

        // Check if the moving voxels are masked out
        float mask_value = 1.0f;
        if (_moving_mask.ptr) {
            mask_value = cuda::linear_at_border<float>(
                    _moving_mask, _moving_dims, moving_p.x, moving_p.y, moving_p.z);

            if (mask_value <= FLT_EPSILON) {
                return;
            }
        }
        
        float c = _impl(
            _fixed,
            _moving,
            _fixed_dims,
            _moving_dims,
            int3{x,y,z},
            moving_p
        );

        reinterpret_cast<float*>(&_cost(x,y,z))[cost_offset] += _weight * mask_value * c; 
    }

    TImpl _impl;

    cuda::VolumePtr<TImpl::VoxelType> _fixed;
    cuda::VolumePtr<TImpl::VoxelType> _moving;
    dim3 _fixed_dims;
    dim3 _moving_dims;

    cuda::VolumePtr<float> _fixed_mask;
    cuda::VolumePtr<float> _moving_mask;
    cuda::VolumePtr<float4> _df;

    float3 _fixed_origin;
    float3 _fixed_spacing;
    Matrix3x3f _fixed_direction;

    float3 _moving_origin;
    float3 _inv_moving_spacing;
    Matrix3x3f _inv_moving_direction;

    float _weight;

    cuda::VolumePtr<float2> _cost;
};

template<typename TKernel>
__global__ void cost_function_kernel(
    TKernel kernel, 
    int3 offset, 
    int3 dims,
    float3 delta,
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

    kernel(x, y, z, delta, cost_offset);
}

template<typename TKernel>
void invoke_cost_function_kernel(
    const TKernel& kernel,
    const float3& delta,
    const int3& offset,
    const int3& dims,
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

    // E(u(x))
    cost_function_kernel<<<grid_size, block_size, 0, stream>>>(
        kernel,
        offset,
        dims,
        float3{0,0,0},
        0
    );

    // E(u(x)+d)
    cost_function_kernel<<<grid_size, block_size, 0, stream>>>(
        kernel,
        offset,
        dims,
        delta,
        1
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}