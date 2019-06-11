#include "landmarks.h"

namespace cuda = stk::cuda;

template<typename T>
__global__ void landmarks_kernel(
    const float3 * const __restrict landmarks,
    const float3 * const __restrict displacements,
    const size_t landmark_count,
    const cuda::VolumePtr<float4> df,
    const float3 delta,
    const float weight,
    const int3 offset,
    const int3 dims,
    const float3 fixed_origin,
    const float3 fixed_spacing,
    const Matrix3x3f fixed_direction,
    cuda::VolumePtr<float> cap_source,
    cuda::VolumePtr<float> cap_sink,
    const float half_decay,
    const float epsilon = 1e-6f
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

    x += offset.x;
    y += offset.y;
    z += offset.z;

    float3 d0 { df(x,y,z).x, df(x,y,z).y, df(x,y,z).z };
    float3 d1 = d0 + delta;

    float3 xyz = float3{float(x),float(y),float(z)};
    float3 world_p = fixed_origin + fixed_direction * (xyz * fixed_spacing);

    for (size_t i = 0; i < landmark_count; ++i) {
        const float inv_den = 1.0f / (pow(stk::norm2(landmarks[i] - world_p), half_decay) + epsilon);
        cap_source(x,y,z) += weight * stk::norm2(d0 - displacements[i]) * inv_den;
        cap_sink(x,y,z) += weight * stk::norm2(d1 - displacements[i]) * inv_den;
    }
}

GpuCostFunction_Landmarks::GpuCostFunction_Landmarks(
        const std::vector<float3>& fixed_landmarks,
        const std::vector<float3>& moving_landmarks,
        const stk::GpuVolume& fixed,
        const float decay)
    : _landmarks {fixed_landmarks}
    , _displacements (fixed_landmarks.size())
    , _fixed {fixed}
    , _half_decay {decay / 2.0f}
{
    ASSERT(fixed_landmarks.size() == moving_landmarks.size());
    for (size_t i = 0; i < fixed_landmarks.size(); ++i) {
        _displacements[i] = moving_landmarks[i] - fixed_landmarks[i];
    }
}

GpuCostFunction_Landmarks::~GpuCostFunction_Landmarks() {}

void GpuCostFunction_Landmarks::cost(
    stk::GpuVolume& df,
    const float3& delta,
    float weight,
    const int3& offset,
    const int3& dims,
    stk::GpuVolume& cap_sink,
    stk::GpuVolume& cap_source,
    stk::cuda::Stream& stream
)
{
    ASSERT(df.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(cap_sink.voxel_type() == stk::Type_Float);
    ASSERT(cap_source.voxel_type() == stk::Type_Float);

    dim3 block_size {32, 32, 1};

    if (dims.x <= 16 || dims.y <= 16) {
        block_size = {16, 16, 4};
    }

    dim3 grid_size {
        (dims.x + block_size.x - 1) / block_size.x,
        (dims.y + block_size.y - 1) / block_size.y,
        (dims.z + block_size.z - 1) / block_size.z
    };

    landmarks_kernel<float><<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(_landmarks.data()),
        thrust::raw_pointer_cast(_displacements.data()),
        _landmarks.size(),
        df,
        delta,
        weight,
        offset,
        dims,
        _fixed.origin(),
        _fixed.spacing(),
        _fixed.direction(),
        cap_sink,
        cap_source,
        _half_decay
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
}

