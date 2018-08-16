#include "transform.h"

#include <stk/common/assert.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/ptr.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/float4.h>

namespace cuda = stk::cuda;

template<typename T>
__global__ void transform_kernel(
    cudaTextureObject_t in,
    dim3 dims,
    cuda::VolumePtr<T> out
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= new_dims.x ||
        y >= new_dims.y ||
        z >= new_dims.z)
    {
        return;
    }

}

stk::GpuVolume gpu::transform_volume(
    const stk::GpuVolume& src, 
    const stk::GpuVolume& def, 
    transform::Interp i = transform::Interp_Linear
)
{
    FATAL_IF(def.voxel_type() != stk::Type_Float4)
        << "Invalid format for displacement";
    
    // TODO: Maybe not neccessary for NN interp
    stk::GpuVolume& src_tex = src.as_usage(stk::gpu::Usage_Texture);

    ASSERT(def.usage() == stk::gpu::Usage_PitchedPointer);



    return stk::Volume();
}
