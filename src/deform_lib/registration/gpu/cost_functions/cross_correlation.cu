#include "cost_function_kernel.h"
#include "cross_correlation.h"

namespace cuda = stk::cuda;

template<typename T>
struct NCCImpl
{
    typedef T VoxelType;

    NCCImpl(int radius) : _radius(radius) {}

    __device__ float operator()(
        const cuda::VolumePtr<VoxelType>& fixed,
        const cuda::VolumePtr<VoxelType>& moving,
        const dim3& fixed_dims,
        const dim3& moving_dims,
        const int3& fixed_p,
        const float3& moving_p,
        const float3& /*d*/
    )
    {
        double sff = 0.0f;
        double sf = 0.0f;

        double smm = 0.0f;
        double sfm = 0.0f;
        double sm = 0.0f;

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

        double denom = sqrt(sff*smm);

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

void GpuCostFunction_NCC::cost(
    GpuDisplacementField& df,
    const float3& delta,
    float weight,
    const int3& offset,
    const int3& dims,
    stk::GpuVolume& cost_acc,
    stk::cuda::Stream& stream
)
{
    ASSERT(cost_acc.voxel_type() == stk::Type_Float2);

    FATAL_IF(_fixed.voxel_type() != stk::Type_Float ||
             _moving.voxel_type() != stk::Type_Float ||
             _fixed_mask.valid() && _fixed_mask.voxel_type() != stk::Type_Float ||
             _moving_mask.valid() && _moving_mask.voxel_type() != stk::Type_Float)
        << "Unsupported pixel type";

    auto kernel = CostFunctionKernel<NCCImpl<float>>(
        NCCImpl<float>(_radius),
        _fixed,
        _moving,
        _fixed_mask,
        _moving_mask,
        weight,
        cost_acc
    );

    invoke_cost_function_kernel(kernel, delta, offset, dims, df, stream);
}

