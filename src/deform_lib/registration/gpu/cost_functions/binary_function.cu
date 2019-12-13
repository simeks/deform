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

// Keep d0 and d1 separate to avoid having to resample them every time
template<typename TDisplacementField>
__device__ void do_step(
    TDisplacementField& df,
    const int3& step,
    const int3& p, // Global coordinates
    const float4& d0,
    const float4& d1,
    const float4& delta,
    float weight,
    float scale,
    float half_exponent,
    float inv_spacing2_exp,
    cuda::VolumePtr<float4>& cost
)
{
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

    cost(p.x,p.y,p.z) = weight*inv_spacing2_exp*e;
}

template<typename TDisplacementField>
__device__ void do_step_border(
    TDisplacementField& df,
    const int3& step,
    const int3& p, // Global coordinates
    const float4& d0,
    const float4& d1,
    const float4& delta,
    float weight,
    float scale,
    float half_exponent,
    float inv_spacing2_exp,
    cuda::VolumePtr<float4>& cost
)
{
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

    int3 p2 = p - step;
    cost(p2.x, p2.y, p2.z).x = weight*inv_spacing2_exp*e.x;
    cost(p2.x, p2.y, p2.z).y = weight*inv_spacing2_exp*e.z;
    cost(p2.x, p2.y, p2.z).z = weight*inv_spacing2_exp*e.x; // border nodes can't move
    cost(p2.x, p2.y, p2.z).w = cost(p2.x, p2.y, p2.z).z;
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
    // Can't add these directly into block_p for some reason
    // Results in "invalid narrowing conversion" even with casting
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int3 block_p {x, y, z};

    if (block_p.x >= dims.x ||
        block_p.y >= dims.y ||
        block_p.z >= dims.z)
    {
        return;
    }

    int3 p {
        block_p.x + offset.x,
        block_p.y + offset.y,
        block_p.z + offset.z
    };

    float4 d0 = df.get(p);
    float4 d1 = df.get(p, delta);

    // Compute energies within the block
    if (p.x + 1 < (int) df.size().x) {
        do_step(df, int3{1, 0, 0}, p, d0, d1, delta, weight, scale, half_exponent,
            inv_spacing2_exp.x, cost_x
        );
    }
    if (p.y + 1 < (int) df.size().y) {
        do_step(df, int3{0, 1, 0}, p, d0, d1, delta, weight, scale, half_exponent,
            inv_spacing2_exp.y, cost_y
        );
    }
    if (p.z + 1 < (int) df.size().z) {
        do_step(df, int3{0, 0, 1}, p, d0, d1, delta, weight, scale, half_exponent,
            inv_spacing2_exp.z, cost_z
        );
    }


    // Compute energies at block border
    if (block_p.x == 0 && p.x != 0) {
        do_step_border(df, int3{-1, 0, 0}, p, d0, d1, delta, weight, scale,
            half_exponent, inv_spacing2_exp.x, cost_x
        );
    }

    if (block_p.y == 0 && p.y != 0) {
        do_step_border(df, int3{0, -1, 0}, p, d0, d1, delta, weight, scale,
            half_exponent, inv_spacing2_exp.y, cost_y
        );
    }

    if (block_p.z == 0 && p.z != 0) {
        do_step_border(df, int3{0, 0, -1}, p, d0, d1, delta, weight, scale,
            half_exponent, inv_spacing2_exp.z, cost_z
        );
    }
}

template<typename TDisplacementField>
__global__ void regularizer_kernel_with_map(
    TDisplacementField df,
    float4 delta,
    cuda::VolumePtr<float> weight_map,
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
    // Can't add these directly into block_p for some reason
    // Results in "invalid narrowing conversion" even with casting
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int3 block_p {x, y, z};

    if (block_p.x >= dims.x ||
        block_p.y >= dims.y ||
        block_p.z >= dims.z)
    {
        return;
    }

    int3 p {
        block_p.x + offset.x,
        block_p.y + offset.y,
        block_p.z + offset.z
    };

    float4 d0 = df.get(p);
    float4 d1 = df.get(p, delta);

    // Compute energies within the block
    if (p.x + 1 < (int) df.size().x) {
        int3 step{1, 0, 0};
        float weight = 0.5f*(weight_map(p.x, p.y, p.z) 
            + weight_map(p.x+step.x, p.y+step.y, p.z+step.z));
        do_step(df, int3{1, 0, 0}, p, d0, d1, delta, weight, scale, half_exponent,
            inv_spacing2_exp.x, cost_x
        );
    }
    if (p.y + 1 < (int) df.size().y) {
        int3 step{0, 1, 0};
        float weight = 0.5f*(weight_map(p.x, p.y, p.z) 
            + weight_map(p.x+step.x, p.y+step.y, p.z+step.z));
        do_step(df, int3{0, 1, 0}, p, d0, d1, delta, weight, scale, half_exponent,
            inv_spacing2_exp.y, cost_y
        );
    }
    if (p.z + 1 < (int) df.size().z) {
        int3 step{0, 0, 1};
        float weight = 0.5f*(weight_map(p.x, p.y, p.z) 
            + weight_map(p.x+step.x, p.y+step.y, p.z+step.z));
        do_step(df, int3{0, 0, 1}, p, d0, d1, delta, weight, scale, half_exponent,
            inv_spacing2_exp.z, cost_z
        );
    }


    // Compute energies at block border
    if (block_p.x == 0 && p.x != 0) {
        int3 step{-1, 0, 0};
        float weight = 0.5f*(weight_map(p.x, p.y, p.z) 
            + weight_map(p.x+step.x, p.y+step.y, p.z+step.z));
        do_step_border(df, int3{-1, 0, 0}, p, d0, d1, delta, weight, scale,
            half_exponent, inv_spacing2_exp.x, cost_x
        );
    }

    if (block_p.y == 0 && p.y != 0) {
        int3 step{0, -1, 0};
        float weight = 0.5f*(weight_map(p.x, p.y, p.z) 
            + weight_map(p.x+step.x, p.y+step.y, p.z+step.z));
        do_step_border(df, int3{0, -1, 0}, p, d0, d1, delta, weight, scale,
            half_exponent, inv_spacing2_exp.y, cost_y
        );
    }

    if (block_p.z == 0 && p.z != 0) {
        int3 step{0, 0, -1};
        float weight = 0.5f*(weight_map(p.x, p.y, p.z) 
            + weight_map(p.x+step.x, p.y+step.y, p.z+step.z));
        do_step_border(df, int3{0, 0, -1}, p, d0, d1, delta, weight, scale,
            half_exponent, inv_spacing2_exp.z, cost_z
        );
    }
}

void GpuBinaryFunction::operator()(
        const GpuDisplacementField& df,
        const float3& delta,
        const int3& offset,
        const int3& dims,
        Settings::UpdateRule update_rule,
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

    if (_weight_map.valid()) {
        if (update_rule == Settings::UpdateRule_Compositive) {
            regularizer_kernel_with_map<cuda::DisplacementField<cuda::CompositiveUpdate>>
            <<<grid_size, block_size, 0, stream>>>(df, delta4, _weight_map, _scale,
                _half_exponent, offset, dims, inv_spacing2_exp, cost_x, cost_y,
                cost_z
            );
        }
        else if (update_rule == Settings::UpdateRule_Additive) {
            regularizer_kernel_with_map<cuda::DisplacementField<cuda::AdditiveUpdate>>
            <<<grid_size, block_size, 0, stream>>>(df, delta4, _weight_map, _scale,
                _half_exponent, offset, dims, inv_spacing2_exp, cost_x, cost_y,
                cost_z
            );
        }
    }
    else {
        if (update_rule == Settings::UpdateRule_Compositive) {
            regularizer_kernel<cuda::DisplacementField<cuda::CompositiveUpdate>>
            <<<grid_size, block_size, 0, stream>>>(df, delta4, _weight, _scale,
                _half_exponent, offset, dims, inv_spacing2_exp, cost_x, cost_y,
                cost_z
            );
        }
        else if (update_rule == Settings::UpdateRule_Additive) {
            regularizer_kernel<cuda::DisplacementField<cuda::AdditiveUpdate>>
            <<<grid_size, block_size, 0, stream>>>(df, delta4, _weight, _scale,
                _half_exponent, offset, dims, inv_spacing2_exp, cost_x, cost_y,
                cost_z
            );
        }
    }

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

