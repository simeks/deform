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
    cudaTextureObject_t src,
    cuda::VolumePtr<float4> def,
    dim3 dims,
    cuda::VolumePtr<T> out
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

    float4 d = def(x,y,z);
    out(x,y,z) = tex3D<T>(src, x+d.x+0.5f, y+d.y+0.5f, z+d.z+0.5f);
}

static void run_kernel(
    stk::Type type,
    const dim3& grid_size,
    const dim3& block_size,
    const cuda::TextureObject& src,
    const stk::GpuVolume& def,
    dim3 dims,
    stk::GpuVolume& out
)
{
    switch (type) {
    case stk::Type_Char:
        transform_kernel<char><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_Char2:
        transform_kernel<char2><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_Char4:
        transform_kernel<char4><<<grid_size, block_size>>>(src, def, dims, out);
        break;

    case stk::Type_UChar:
        transform_kernel<uint8_t><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_UChar2:
        transform_kernel<uchar2><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_UChar4:
        transform_kernel<uchar4><<<grid_size, block_size>>>(src, def, dims, out);
        break;

    case stk::Type_Short:
        transform_kernel<short><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_Short2:
        transform_kernel<short2><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_Short4:
        transform_kernel<short4><<<grid_size, block_size>>>(src, def, dims, out);
        break;

    case stk::Type_UShort:
        transform_kernel<uint16_t><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_UShort2:
        transform_kernel<ushort2><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_UShort4:
        transform_kernel<ushort4><<<grid_size, block_size>>>(src, def, dims, out);
        break;

    case stk::Type_Int:
        transform_kernel<int><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_Int2:
        transform_kernel<int2><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_Int4:
        transform_kernel<int4><<<grid_size, block_size>>>(src, def, dims, out);
        break;

    case stk::Type_UInt:
        transform_kernel<uint32_t><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_UInt2:
        transform_kernel<uint2><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_UInt4:
        transform_kernel<uint4><<<grid_size, block_size>>>(src, def, dims, out);
        break;

    case stk::Type_Float:
        transform_kernel<float><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_Float2:
        transform_kernel<float2><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    case stk::Type_Float4:
        transform_kernel<float4><<<grid_size, block_size>>>(src, def, dims, out);
        break;
    default:
        FATAL() << "Unsupported format";
    };
}


stk::GpuVolume gpu::transform_volume(
    const stk::GpuVolume& src, 
    const stk::GpuVolume& def, 
    transform::Interp i
)
{
    ASSERT(def.usage() == stk::gpu::Usage_PitchedPointer);
    FATAL_IF(def.voxel_type() != stk::Type_Float4)
        << "Invalid format for displacement";
    
    // PERF? Maybe not neccessary for NN interp
    stk::GpuVolume src_tex = src.as_usage(stk::gpu::Usage_Texture);

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeBorder;
    tex_desc.addressMode[1] = cudaAddressModeBorder;
    tex_desc.addressMode[2] = cudaAddressModeBorder;
    tex_desc.filterMode = i == transform::Interp_Linear ? cudaFilterModeLinear : cudaFilterModePoint;

    dim3 dims = def.size();

    stk::GpuVolume out(dims, src.voxel_type());
    out.copy_meta_from(def);

    dim3 block_size{32,32,1};
    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    run_kernel(src.voxel_type(), grid_size, block_size, 
        cuda::TextureObject(src_tex, tex_desc), def, dims, out);

    // PERF?
    return out.as_usage(src.usage());
}
