#include "cost_function_kernel.h"
#include "soft_constraints.h"

#include "../gpu_displacement_field.h"

#include <stk/math/float4.h>

namespace cuda {
    using namespace stk::cuda;
}

template<typename T>
struct SoftConstraintsImpl
{
    typedef T VoxelType;

    SoftConstraintsImpl(
        const cuda::VolumePtr<uint8_t>& constraint_mask,
        const cuda::VolumePtr<float4>& constraint_values
    ) : _constraint_mask(constraint_mask), _constraint_values(constraint_values)
    {}

    __device__ float operator()(
        const cuda::VolumePtr<VoxelType>& fixed,
        const cuda::VolumePtr<VoxelType>& moving,
        const dim3& /*fixed_dims*/,
        const dim3& /*moving_dims*/,
        const int3& fixed_p,
        const float3& /*moving_p*/,
        const float3& d
    )
    {
        float4 diff = float4{d.x, d.y, d.z, 0.0f} - _constraint_values(fixed_p.x, fixed_p.y, fixed_p.z);
        float dist2 = fminf(stk::norm2(diff), 100000.0f); // Clamp to avoid explosion
        return float(_constraint_mask(fixed_p.x, fixed_p.y, fixed_p.z)) * dist2;
    }

    cuda::VolumePtr<uint8_t> _constraint_mask;
    cuda::VolumePtr<float4> _constraint_values;
};

void GpuCostFunction_SoftConstraints::cost(
    GpuDisplacementField& df,
    const float3& delta,
    float weight,
    const int3& offset,
    const int3& dims,
    Settings::UpdateRule update_rule,
    stk::GpuVolume& cost_acc,
    stk::cuda::Stream& stream
)
{
    ASSERT(cost_acc.voxel_type() == stk::Type_Float2);

    FATAL_IF(_constraint_mask.voxel_type() != stk::Type_UChar ||
             _constraint_values.voxel_type() != stk::Type_Float4)
        << "Unsupported pixel type";

    auto kernel = CostFunctionKernel<SoftConstraintsImpl<float>>(
        SoftConstraintsImpl<float>(
            _constraint_mask,
            _constraint_values
        ),
        // This cost function isn't using either fixed nor moving volume
        stk::GpuVolume(),
        stk::GpuVolume(),
        _fixed_mask,
        _moving_mask,
        weight,
        cost_acc
    );

    invoke_cost_function_kernel(
        kernel,
        delta,
        offset,
        dims,
        df,
        update_rule,
        stream
    );
}

