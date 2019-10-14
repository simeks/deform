#include "landmarks.h"

#include "cost_function_kernel.h"

namespace cuda = stk::cuda;

template<typename T>
struct LandmarksImpl
{
    typedef T VoxelType;

    LandmarksImpl(
        float3 origin,
        float3 spacing,
        Matrix3x3f direction,
        const thrust::device_vector<float4>& landmarks,
        const thrust::device_vector<float4>& displacements,
        float half_decay
    ) :
        _origin(origin),
        _spacing(spacing),
        _direction(direction),
        _landmarks(thrust::raw_pointer_cast(landmarks.data())),
        _displacements(thrust::raw_pointer_cast(displacements.data())),
        _landmark_count(landmarks.size()),
        _half_decay(half_decay)
    {
    }

    __device__ float operator()(
        const cuda::VolumePtr<VoxelType>& fixed,
        const cuda::VolumePtr<VoxelType>& moving,
        const dim3& fixed_dims,
        const dim3& moving_dims,
        const int3& fixed_p,
        const float3& moving_p,
        const float3& d
    )
    {
        const float epsilon = 1e-6f;

        float3 xyz = float3{float(fixed_p.x),float(fixed_p.y),float(fixed_p.z)};
        float3 world_p = _origin + _direction * (xyz * _spacing);

        float c = 0;
        for (size_t i = 0; i < _landmark_count; ++i) {
            float3 lm = { _landmarks[i].x, _landmarks[i].y, _landmarks[i].z };
            float3 dp = { _displacements[i].x, _displacements[i].y, _displacements[i].z };

            const float inv_den = 1.0f 
                / (pow(stk::norm2(lm - world_p), _half_decay) + epsilon);
            c += stk::norm2(d - dp) * inv_den;
        }
        return c;
    }

    float3 _origin;
    float3 _spacing;
    Matrix3x3f _direction;

    const float4 * const __restrict _landmarks;
    const float4 * const __restrict _displacements;
    const size_t _landmark_count;

    const float _half_decay;
};

namespace {
    float4 pad(const float3& v) {
        return float4{v.x,v.y,v.z,0};
    }
}

GpuCostFunction_Landmarks::GpuCostFunction_Landmarks(
    const stk::GpuVolume& fixed,
    const std::vector<float3>& fixed_landmarks,
    const std::vector<float3>& moving_landmarks,
    const float decay
) :
    _origin(fixed.origin()),
    _spacing(fixed.spacing()),
    _direction(fixed.direction()),
    _half_decay(decay / 2.0f)
{
    std::vector<float4> displacements;
    std::vector<float4> landmarks;

    ASSERT(fixed_landmarks.size() == moving_landmarks.size());
    for (size_t i = 0; i < fixed_landmarks.size(); ++i) {
        landmarks.push_back(pad(fixed_landmarks[i]));
        displacements.push_back(pad(moving_landmarks[i] - fixed_landmarks[i]));
    }
    _landmarks = landmarks;
    _displacements = displacements;
}
GpuCostFunction_Landmarks::~GpuCostFunction_Landmarks()
{
}
void GpuCostFunction_Landmarks::cost(
    stk::GpuVolume& df,
    const float3& delta,
    float weight,
    const int3& offset,
    const int3& dims,
    stk::GpuVolume& cost_acc,
    Settings::UpdateRule update_rule,
    stk::cuda::Stream& stream
)
{
    ASSERT(df.usage() == stk::gpu::Usage_PitchedPointer);
    ASSERT(cost_acc.voxel_type() == stk::Type_Float2);

    // <float> isn't really necessary but it is required by CostFunctionKernel
    auto kernel = CostFunctionKernel<LandmarksImpl<float>>(
        LandmarksImpl<float>(
            _origin,
            _spacing,
            _direction,
            _landmarks,
            _displacements,
            _half_decay
        ),
        stk::GpuVolume(),
        stk::GpuVolume(),
        _fixed_mask,
        _moving_mask,
        df,
        weight,
        cost_acc
    );

    invoke_cost_function_kernel(kernel, delta, offset, dims, update_rule, stream);
}


