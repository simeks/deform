#pragma once
#pragma once

#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/matrix3x3f.h>

#include "../settings.h"

/** Wrapper for a displacement field image.
 * 
 * Users are adviced to use the transform_* methods, rather then applying the
 * displacements themselves via `df.get(p)`.
 * */
class GpuDisplacementField
{
public:
    GpuDisplacementField() : _update_rule(Settings::UpdateRule_Additive) {}
    GpuDisplacementField(
        const stk::GpuVolume& df,
        Settings::UpdateRule update_rule = Settings::UpdateRule_Additive
    ) :
        _update_rule(update_rule),
        _df(df)
    {
    }
    ~GpuDisplacementField() {}

    GpuDisplacementField clone()
    {
        return GpuDisplacementField(_df.clone(), _update_rule);
    }

    void copy_from(const GpuDisplacementField& other)
    {
        _df.copy_from(other.volume());
    }

    dim3 size() const
    {
        return _df.size();
    }

    // Volume containing the displacements only
    const stk::GpuVolume& volume() const
    {
        return _df;
    }

    const float3& origin() const
    {
        return _df.origin();
    }

    const float3& spacing() const
    {
        return _df.spacing();
    }
    
    const Matrix3x3f& direction() const
    {
        return _df.direction();
    }

    Settings::UpdateRule update_rule() const
    {
        return _update_rule;
    }

private:
    Settings::UpdateRule _update_rule;
    stk::GpuVolume _df;

};

#ifdef __CUDACC__

namespace cuda {

struct CompositiveUpdate
{
    __device__ float4 operator()(
        const stk::cuda::VolumePtr<float4>& df,
        const dim3& dims,
        const float3& inv_spacing,
        int x, int y, int z,
        const float4& delta
    ) {
        // Convert delta from mm to image space
        return stk::cuda::linear_at_clamp<float4>(
            df,
            dims,
            x + delta.x * inv_spacing.x,
            y + delta.y * inv_spacing.y,
            z + delta.z * inv_spacing.z
        ) + delta;
    }
};

struct AdditiveUpdate
{
    __device__ float4 operator()(
        const stk::cuda::VolumePtr<float4>& df,
        const dim3& /*dims*/,
        const float3& /*inv_spacing*/,
        int x, int y, int z,
        const float4& delta
    ) {
        return df(x, y, z) + delta;
    }
};

template<typename TUpdateFn = AdditiveUpdate>
class DisplacementField
{
public:
    DisplacementField(const GpuDisplacementField& df) :
        _df(df.volume()),
        _dims(df.size()),
        _origin(df.origin()),
        _spacing(df.spacing()),
        _direction(df.direction())
    {
        _inv_spacing = {
            1.0f / _spacing.x,
            1.0f / _spacing.y,
            1.0f / _spacing.z
        };
    }
    ~DisplacementField() {}

    __device__ float4 get(const int3& p) const
    {
        return _df(p.x, p.y, p.z);
    }

    __device__ float4 get(const int3& p, const float4& delta) const
    {
        TUpdateFn fn;
        return fn(_df, _dims, _inv_spacing, p.x, p.y, p.z, delta);
    }

    // p : Index in displacement field
    // Returns coordinates in world space
    __device__ float3 transform_index(const int3& p) const
    {
        float4 d = get(p);
        return index2point(p) + float3{d.x, d.y, d.z};
    }

    __device__ float3 index2point(const int3& index) const
    {
        float3 xyz {float(index.x), float(index.y), float(index.z)};
        return _origin + _direction * (_spacing * xyz);
    }

    __device__ const dim3& size() const
    {
        return _dims;
    }

private:
    stk::cuda::VolumePtr<float4> _df;
    dim3 _dims;

    float3 _origin;
    float3 _spacing;
    float3 _inv_spacing;
    Matrix3x3f _direction;
};
} // namespace cuda
#endif // __CUDACC__
