#include "gpu_displacement_field.h"

#include <stk/cuda/cuda.h>
#include <stk/cuda/volume.h>

namespace cuda {
    using namespace stk::cuda;
}

__global__ void compute_displacement_field_kernel(
    cuda::DisplacementField<> df_in,
    cuda::VolumePtr<float4> df_out
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= df_in.size().x ||
        y >= df_in.size().y ||
        z >= df_in.size().z)
    {
        return;
    }
    df_out(x, y, z) = df_in.get(int3{x, y, z});
}

stk::GpuVolume cuda::compute_displacement_field(
    const stk::GpuVolume& vector_field,
    const AffineTransform& affine
)
{
    ASSERT(vector_field.voxel_type() == stk::Type_Float4);

    stk::GpuVolume out(vector_field.size(), stk::Type_Float4);
    out.copy_meta_from(vector_field);

    dim3 dims = vector_field.size();
    dim3 block_size {32,32,1};
    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };
    GpuDisplacementField df(vector_field, affine);
    compute_displacement_field_kernel<<<grid_size, block_size>>>(
        cuda::DisplacementField<>(df),
        out
    );
    return out;
}

