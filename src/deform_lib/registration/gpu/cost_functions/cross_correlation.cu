#include "cross_correlation.h"

#include <float.h>

namespace cuda = stk::cuda;

template<typename T, bool use_fixed_mask, bool use_moving_mask>
__global__ void ncc_kernel(
    cuda::VolumePtr<T> fixed,
    cuda::VolumePtr<T> moving,
    const cuda::VolumePtr<float> fixed_mask,
    const cuda::VolumePtr<float> moving_mask,
    cuda::VolumePtr<float4> df,
    float3 delta,
    float weight,
    int radius,
    int3 offset,
    int3 dims,
    dim3 fixed_dims,
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

    const float3 d0 { df(x,y,z).x, df(x, y, z).y, df(x, y, z).z };
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

        if (mask_value_0 < FLT_EPSILON && mask_value_1 < FLT_EPSILON) {
            return;
        }
    }

    float sff = 0.0f;
    float sf = 0.0f;

    float smm0 = 0.0f;
    float smm1 = 0.0f;
    float sfm0 = 0.0f;
    float sfm1 = 0.0f;
    float sm0 = 0.0f;
    float sm1 = 0.0f;

    unsigned int n = 0;

    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                // TODO: Does not account for anisotropic volumes
                int r2 = dx*dx + dy*dy + dz*dz;
                if (r2 > radius * radius)
                    continue;

                int3 fp{x + dx, y + dy, z + dz};

                if (fp.x < 0 || fp.x >= int(fixed_dims.x) ||
                    fp.y < 0 || fp.y >= int(fixed_dims.y) ||
                    fp.z < 0 || fp.z >= int(fixed_dims.z))
                    continue;

                float3 mp0{moving_p0.x + dx, moving_p0.y + dy, moving_p0.z + dz};
                float3 mp1{moving_p1.x + dx, moving_p1.y + dy, moving_p1.z + dz};

                float fixed_v = fixed(fp.x, fp.y, fp.z);

                float moving_v0 = cuda::linear_at_border<float>(moving, moving_dims, mp0.x, mp0.y, mp0.z);
                float moving_v1 = cuda::linear_at_border<float>(moving, moving_dims, mp1.x, mp1.y, mp1.z);

                sff += fixed_v * fixed_v;

                smm0 += moving_v0 * moving_v0;
                smm1 += moving_v1 * moving_v1;

                sfm0 += fixed_v*moving_v0;
                sfm1 += fixed_v*moving_v1;

                sm0 += moving_v0;
                sm1 += moving_v1;

                sf += fixed_v;

                ++n;
            }
        }
    }

    if (n == 0)
        return;

    // Subtract mean
    sff -= (sf * sf / n);
    smm0 -= (sm0 * sm0 / n);
    sfm0 -= (sf * sm0 / n);
    smm1 -= (sm1 * sm1 / n);
    sfm1 -= (sf * sm1 / n);

    float denom1 = sqrt(sff*smm1);
    float denom0 = sqrt(sff*smm0);

    // Set cost to zero if outside moving volume

    if (moving_p0.x >= 0 && moving_p0.x < moving_dims.x &&
        moving_p0.y >= 0 && moving_p0.y < moving_dims.y &&
        moving_p0.z >= 0 && moving_p0.z < moving_dims.z &&
        denom0 > 1e-5 && mask_value_0 >= FLT_EPSILON)
    {
        cost_acc(x,y,z).x += mask_value_0 * weight * 0.5f * (1.0f-float(sfm0 / denom0));
    }

    if (moving_p1.x >= 0 && moving_p1.x < moving_dims.x &&
        moving_p1.y >= 0 && moving_p1.y < moving_dims.y &&
        moving_p1.z >= 0 && moving_p1.z < moving_dims.z &&
        denom1 > 1e-5 && mask_value_1 >= FLT_EPSILON)
    {
        cost_acc(x,y,z).y += mask_value_1 * weight * 0.5f * (1.0f-float(sfm1 / denom1));
    }
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

    auto (*kernel) = &ncc_kernel<float, false, false>;
    if (_fixed_mask.valid()) {
        if (_moving_mask.valid()) {
            kernel = &ncc_kernel<float, true, true>;
        }
        else {
            kernel = &ncc_kernel<float, true, false>;
        }
    }
    else {
        if (_moving_mask.valid()) {
            kernel = &ncc_kernel<float, false, true>;
        }
        else {
            kernel = &ncc_kernel<float, false, false>;
        }
    }

    kernel<<<grid_size, block_size, 0, stream>>>(
        _fixed,
        _moving,
        _fixed_mask,
        _moving_mask,
        df,
        delta,
        weight,
        _radius,
        offset,
        dims,
        _fixed.size(),
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

