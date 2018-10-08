#include "transform.h"

#include <stk/common/assert.h>
#include <stk/cuda/cuda.h>
#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/float4.h>

namespace cuda = stk::cuda;

template<typename T>
__global__ void transform_kernel_linear(
    cuda::VolumePtr<T> src,
    dim3 src_dims,
    cuda::VolumePtr<float4> df,
    dim3 df_dims,
    float3 fixed_origin,
    float3 fixed_spacing,
    Matrix3x3f fixed_direction,
    float3 moving_origin,
    float3 inv_moving_spacing,
    Matrix3x3f inv_moving_direction,
    cuda::VolumePtr<T> out
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= df_dims.x ||
        y >= df_dims.y ||
        z >= df_dims.z)
    {
        return;
    }

    float3 xyz = float3{float(x), float(y), float(z)};

    float3 wp = fixed_origin + fixed_direction * (xyz * fixed_spacing);

    float4 d = df(x,y,z);
    float3 mp = (inv_moving_direction * (wp + float3{d.x, d.y, d.z} - moving_origin))
                * inv_moving_spacing;

    out(x,y,z) = cuda::linear_at_border(src, src_dims, mp.x, mp.y, mp.z);
}

template<typename T>
__global__ void transform_kernel_nn(
    cuda::VolumePtr<T> src,
    dim3 src_dims,
    cuda::VolumePtr<float4> df,
    dim3 df_dims,
    float3 fixed_origin,
    float3 fixed_spacing,
    Matrix3x3f fixed_direction,
    float3 moving_origin,
    float3 inv_moving_spacing,
    Matrix3x3f inv_moving_direction,
    cuda::VolumePtr<T> out
)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= df_dims.x ||
        y >= df_dims.y ||
        z >= df_dims.z)
    {
        return;
    }

    float3 xyz = float3{float(x), float(y), float(z)};

    float3 wp = fixed_origin + fixed_direction * (xyz * fixed_spacing);

    float4 d = df(x,y,z);
    float3 mp = (inv_moving_direction * (wp + float3{d.x, d.y, d.z} - moving_origin))
                * inv_moving_spacing;

    int xt = roundf(mp.x);
    int yt = roundf(mp.y);
    int zt = roundf(mp.z);

    if (xt >= 0 && xt < src_dims.x &&
        yt >= 0 && yt < src_dims.y &&
        zt >= 0 && zt < src_dims.z) {

        out(x,y,z) = src(xt, yt, zt);
    }
    else {
        out(x,y,z) = T{0};
    }
}

static void run_nn_kernel(
    stk::Type type,
    const dim3& grid_size,
    const dim3& block_size,
    const stk::GpuVolume& src,
    const stk::GpuVolume& df,
    stk::GpuVolume& out
)
{
    float3 inv_moving_spacing = float3{1.0f, 1.0f, 1.0f} / src.spacing();

    #define TRANSFORM_KERNEL_NN(type) \
            transform_kernel_nn<type><<<grid_size, block_size>>>( \
                src, \
                src.size(), \
                df, \
                df.size(), \
                df.origin(), \
                df.spacing(), \
                df.direction(), \
                src.origin(), \
                inv_moving_spacing, \
                src.inverse_direction(), \
                out \
            )

    switch (type) {
    case stk::Type_Char:  TRANSFORM_KERNEL_NN(char);  break;
    case stk::Type_Char2: TRANSFORM_KERNEL_NN(char2); break;
    case stk::Type_Char4: TRANSFORM_KERNEL_NN(char4); break;

    case stk::Type_UChar:  TRANSFORM_KERNEL_NN(uint8_t); break;
    case stk::Type_UChar2: TRANSFORM_KERNEL_NN(uchar2);  break;
    case stk::Type_UChar4: TRANSFORM_KERNEL_NN(uchar4);  break;

    case stk::Type_Short:  TRANSFORM_KERNEL_NN(short);  break;
    case stk::Type_Short2: TRANSFORM_KERNEL_NN(short2); break;
    case stk::Type_Short4: TRANSFORM_KERNEL_NN(short4); break;

    case stk::Type_UShort:  TRANSFORM_KERNEL_NN(uint16_t); break;
    case stk::Type_UShort2: TRANSFORM_KERNEL_NN(ushort2);  break;
    case stk::Type_UShort4: TRANSFORM_KERNEL_NN(ushort4);  break;

    case stk::Type_Int:  TRANSFORM_KERNEL_NN(int);  break;
    case stk::Type_Int2: TRANSFORM_KERNEL_NN(int2); break;
    case stk::Type_Int4: TRANSFORM_KERNEL_NN(int4); break;

    case stk::Type_UInt:  TRANSFORM_KERNEL_NN(uint32_t); break;
    case stk::Type_UInt2: TRANSFORM_KERNEL_NN(uint2);    break;
    case stk::Type_UInt4: TRANSFORM_KERNEL_NN(uint4);    break;

    case stk::Type_Float:  TRANSFORM_KERNEL_NN(float);  break;
    case stk::Type_Float2: TRANSFORM_KERNEL_NN(float2); break;
    case stk::Type_Float4: TRANSFORM_KERNEL_NN(float4); break;

    default:
        FATAL() << "Unsupported pixel type";
    };

    #undef TRANSFORM_KERNEL_NN
}

static void run_linear_kernel(
    stk::Type type,
    const dim3& grid_size,
    const dim3& block_size,
    const stk::GpuVolume& src,
    const stk::GpuVolume& df,
    stk::GpuVolume& out
)
{
    #define TRANSFORM_KERNEL_LINEAR(type) \
            transform_kernel_linear<type><<<grid_size, block_size>>>( \
                src, \
                src.size(), \
                df, \
                df.size(), \
                df.origin(), \
                df.spacing(), \
                df.direction(), \
                src.origin(), \
                inv_moving_spacing, \
                src.inverse_direction(), \
                out \
            )

    float3 inv_moving_spacing = float3{1.0f, 1.0f, 1.0f} / src.spacing();

    switch (type) {

    case stk::Type_Float:  TRANSFORM_KERNEL_LINEAR(float);  break;
    case stk::Type_Float2: TRANSFORM_KERNEL_LINEAR(float2); break;
    case stk::Type_Float4: TRANSFORM_KERNEL_LINEAR(float4); break;

    default:
        FATAL() << "Interpolation mode only supports float types";
    };

    #undef TRANSFORM_KERNEL_LINEAR
}


stk::GpuVolume gpu::transform_volume(
    const stk::GpuVolume& src,
    const stk::GpuVolume& def,
    transform::Interp i,
    const dim3& block_size
)
{
    ASSERT(def.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(src.usage() == stk::gpu::Usage_PitchedPointer);
    FATAL_IF(def.voxel_type() != stk::Type_Float4)
        << "Invalid pixel type for displacement";

    dim3 dims = def.size();

    stk::GpuVolume out(dims, src.voxel_type());
    out.copy_meta_from(def);

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    if (i == transform::Interp_NN) {
        run_nn_kernel(src.voxel_type(), grid_size, block_size, src, def, out);
    } else {
        run_linear_kernel(src.voxel_type(), grid_size, block_size, src, def, out);
    }

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

    return out;
}
