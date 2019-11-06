#include "binary_function.h"
#include "cost_function_kernel.h"

#include <stk/math/float4.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/stream.h>
#include <stk/cuda/volume.h>


namespace cuda {
    using namespace stk::cuda;
}

__device__ float4 energy(
    float4 d0,
    float4 d1,
    float4 dn0,
    float4 dn1,
    float scale,
    float half_exponent)
{
    return {
        pow(scale * stk::norm2(d0 - dn0), half_exponent),
        pow(scale * stk::norm2(d0 - dn1), half_exponent),
        pow(scale * stk::norm2(d1 - dn0), half_exponent),
        pow(scale * stk::norm2(d1 - dn1), half_exponent)
    };
}

template<typename TDisplacementField>
__global__ void regularizer_kernel(
    TDisplacementField df,
    float4 delta,
    float weight,
    float scale,
    float half_exponent,
    int3 offset,
    int3 dims,
    float3 inv_spacing2_exp,
    cuda::VolumePtr<float4> cost_x, // Regularization cost in x+
    cuda::VolumePtr<float4> cost_y, // y+
    cuda::VolumePtr<float4> cost_z  // z+
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

    int gx = x + offset.x;
    int gy = y + offset.y;
    int gz = z + offset.z;

    int3 p {gx, gy, gz};

    // Cost ordered as E00, E01, E10, E11

    float4 d0 = df.get(p);
    float4 d1 = df.get(p, delta);

    float4 o_x = {0, 0, 0, 0};
    float4 o_y = {0, 0, 0, 0};
    float4 o_z = {0, 0, 0, 0};

    if (gx + 1 < (int) df.size().x) {
        int3 step {1, 0, 0};
        float4 dn0 = df.get(p+step);
        float4 dn1 = df.get(p+step, delta);

        o_x = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );
    }
    if (gy + 1 < (int) df.size().y) {
        int3 step {0, 1, 0};
        float4 dn0 = df.get(p+step);
        float4 dn1 = df.get(p+step, delta);

        o_y = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );
    }
    if (gz + 1 < (int) df.size().z) {
        int3 step {0, 0, 1};
        float4 dn0 = df.get(p+step);
        float4 dn1 = df.get(p+step, delta);

        o_z = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );
    }
    cost_x(gx,gy,gz) = weight*inv_spacing2_exp.x*o_x;
    cost_y(gx,gy,gz) = weight*inv_spacing2_exp.y*o_y;
    cost_z(gx,gy,gz) = weight*inv_spacing2_exp.z*o_z;


     // Compute cost at block border

    if (x == 0 && gx != 0) {
        int3 step {-1, 0, 0};
        float4 dn0 = df.get(p+step);
        float4 dn1 = df.get(p+step, delta);

        float4 e = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );

        // Here we need to think in reverse, since this are the costs for the 
        //  neighbouring node. I.e. E01 => E10

        cost_x(gx-1,gy,gz).x = weight*inv_spacing2_exp.x*e.x;
        cost_x(gx-1,gy,gz).y = weight*inv_spacing2_exp.x*e.z;
        cost_x(gx-1,gy,gz).z = weight*inv_spacing2_exp.x*e.x; // border nodes can't move
        cost_x(gx-1,gy,gz).w = cost_x(gx-1,gy,gz).z;
    }

    if (y == 0 && gy != 0) {
        int3 step {0, -1, 0};
        float4 dn0 = df.get(p+step);
        float4 dn1 = df.get(p+step, delta);

        float4 e = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );

        cost_y(gx,gy-1,gz).x = weight*inv_spacing2_exp.y*e.x;
        cost_y(gx,gy-1,gz).y = weight*inv_spacing2_exp.y*e.z;
        cost_y(gx,gy-1,gz).z = weight*inv_spacing2_exp.y*e.x; // border nodes can't move
        cost_y(gx,gy-1,gz).w = cost_y(gx,gy-1,gz).z;
    }

    if (z == 0 && gz != 0) {
        int3 step {0, 0, -1};
        float4 dn0 = df.get(p+step);
        float4 dn1 = df.get(p+step, delta);

        float4 e = energy(
            d0,
            d1,
            dn0,
            dn1,
            scale,
            half_exponent
        );

        cost_z(gx,gy,gz-1).x = weight*inv_spacing2_exp.z*e.x;
        cost_z(gx,gy,gz-1).y = weight*inv_spacing2_exp.z*e.z;
        cost_z(gx,gy,gz-1).z = weight*inv_spacing2_exp.z*e.x; // border nodes can't move
        cost_z(gx,gy,gz-1).w = cost_z(gx,gy,gz-1).z;
    }
}

void GpuBinaryFunction::operator()(
        const GpuDisplacementField& df,
        const float3& delta,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cost_x,
        stk::GpuVolume& cost_y,
        stk::GpuVolume& cost_z,
        stk::cuda::Stream& stream
        )
{
    ASSERT(cost_x.voxel_type() == stk::Type_Float4);
    ASSERT(cost_y.voxel_type() == stk::Type_Float4);
    ASSERT(cost_z.voxel_type() == stk::Type_Float4);

    // The binary function is quite register heavy so we need
    // to restrict the thread count (512 rather than 1024).
    dim3 block_size {32, 16, 1};
    if (dims.x <= 16 || dims.y <= 16) {
        block_size = {16, 16, 2};
    }

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    float3 inv_spacing2_exp {
        1.0f / pow(_spacing.x*_spacing.x, _half_exponent),
        1.0f / pow(_spacing.y*_spacing.y, _half_exponent),
        1.0f / pow(_spacing.z*_spacing.z, _half_exponent)
    };

    float4 delta4 {
        delta.x,
        delta.y,
        delta.z,
        0
    };

    if (df.update_rule() == Settings::UpdateRule_Compositive) {
        regularizer_kernel<cuda::DisplacementField<cuda::CompositiveUpdate>>
        <<<grid_size, block_size, 0, stream>>>(
            df,
            delta4,
            _weight,
            _scale,
            _half_exponent,
            offset,
            dims,
            inv_spacing2_exp,
            cost_x,
            cost_y,
            cost_z
        );
    }
    else if (df.update_rule() == Settings::UpdateRule_Additive) {
        regularizer_kernel<cuda::DisplacementField<cuda::AdditiveUpdate>>
        <<<grid_size, block_size, 0, stream>>>(
            df,
            delta4,
            _weight,
            _scale,
            _half_exponent,
            offset,
            dims,
            inv_spacing2_exp,
            cost_x,
            cost_y,
            cost_z
        );
    }
    else {
        ASSERT(false);
    }

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

