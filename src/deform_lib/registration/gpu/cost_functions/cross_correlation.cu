#include "cross_correlation.h"

#include <float.h>

namespace cuda = stk::cuda;

template<typename T, typename TImpl, bool use_fixed_mask=false, bool use_moving_mask=false>
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
        if (use_fixed_mask) {
            if (_fixed_mask(x, y, z) <= FLT_EPSILON) {
                return;
            }
        }

        const float3 d0 { _df(x,y,z).x + delta.x, _df(x, y, z).y + delta.y, _df(x, y, z).z + delta.z };

        const float3 xyz = float3{float(x),float(y),float(z)};
        const float3 world_p = _fixed_origin + _fixed_direction * (xyz * _fixed_spacing);

        const float3 moving_p = (_inv_moving_direction * (world_p + d0 - _moving_origin)) * _inv_moving_spacing;

        // Check if the moving voxels are masked out
        float mask_value = 1.0f;
        if (use_moving_mask) {
            mask_value = cuda::linear_at_border<float>(
                    _moving_mask, _moving_dims, moving_p.x, moving_p.y, moving_p.z);

            if (mask_value < FLT_EPSILON) {
                return;
            }
        }
        
        float c = _impl(
            _fixed,
            _fixed_dims,
            _moving,
            _moving_dims,
            int3{x,y,z},
            moving_p
        );

        reinterpret_cast<float*>(&_cost(x,y,z))[cost_offset] += _weight * mask_value * c; 
    }

    TImpl _impl;

    cuda::VolumePtr<T> _fixed;
    cuda::VolumePtr<T> _moving;
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

template<typename T>
struct NCCImpl
{
    NCCImpl(int radius) : _radius(radius) {}

    __device__ float operator()(
        const cuda::VolumePtr<T>& fixed,
        const dim3& fixed_dims,
        cuda::VolumePtr<T> moving,
        const dim3& moving_dims,
        const int3& fixed_p, // Reference space
        const float3& moving_p // Moving space
    )
    {
        float sff = 0.0f;
        float sf = 0.0f;

        float smm = 0.0f;
        float sfm = 0.0f;
        float sm = 0.0f;

        unsigned int n = 0;

        for (int dz = -_radius; dz <= _radius; ++dz) {
            for (int dy = -_radius; dy <= _radius; ++dy) {
                for (int dx = -_radius; dx <= _radius; ++dx) {
                    // TODO: Does not account for anisotropic volumes
                    int r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 > _radius * _radius)
                        continue;

                    int3 fp{fixed_p.x + dx, fixed_p.y + dy, fixed_p.z + dz};

                    if (fp.x < 0 || fp.x >= int(fixed_dims.x) ||
                        fp.y < 0 || fp.y >= int(fixed_dims.y) ||
                        fp.z < 0 || fp.z >= int(fixed_dims.z))
                        continue;

                    float3 mp{moving_p.x + dx, moving_p.y + dy, moving_p.z + dz};

                    float fixed_v = fixed(fp.x, fp.y, fp.z);

                    float moving_v = cuda::linear_at_border<float>(moving, moving_dims, mp.x, mp.y, mp.z);

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
            return 0;

        // Subtract mean
        sff -= (sf * sf / n);
        smm -= (sm * sm / n);
        sfm -= (sf * sm / n);

        float denom = sqrt(sff*smm);

        // Set cost to zero if outside moving volume

        if (moving_p.x >= 0 && moving_p.x < moving_dims.x &&
            moving_p.y >= 0 && moving_p.y < moving_dims.y &&
            moving_p.z >= 0 && moving_p.z < moving_dims.z &&
            denom > 1e-5)
        {
            return 0.5f * (1.0f-float(sfm / denom));
        }
        return 0;
    }

    int _radius;
};

template<typename TKernel>
__global__ void run_cost_kernel(
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

    // auto (*kernel) = &ncc_kernel<float, false, false>;
    // if (_fixed_mask.valid()) {
    //     if (_moving_mask.valid()) {
    //         kernel = &ncc_kernel<float, true, true>;
    //     }
    //     else {
    //         kernel = &ncc_kernel<float, true, false>;
    //     }
    // }
    // else {
    //     if (_moving_mask.valid()) {
    //         kernel = &ncc_kernel<float, false, true>;
    //     }
    //     else {
    //         kernel = &ncc_kernel<float, false, false>;
    //     }
    // }

    auto kernel = CostFunctionKernel<float,  NCCImpl<float>>(
        NCCImpl<float>(2),
        _fixed,
        _moving,
        _fixed_mask,
        _moving_mask,
        df,
        weight,
        cost_acc
    );

    // E(u(x))
    run_cost_kernel<<<grid_size, block_size, 0, stream>>>(
        kernel,
        offset,
        dims,
        float3{0,0,0},
        0
    );

    // E(u(x)+d)
    run_cost_kernel<<<grid_size, block_size, 0, stream>>>(
        kernel,
        offset,
        dims,
        delta,
        1
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

