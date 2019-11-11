#pragma once
#pragma once

#include <stk/cuda/volume.h>
#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>
#include <stk/math/matrix3x3f.h>

#include "../affine_transform.h"
#include "../settings.h"

/** Wrapper for a displacement field image.
 * 
 * Users are adviced to use the transform_* methods, rather then applying the
 * displacements themselves via `df.get(p)`.
 * */
class GpuDisplacementField
{
public:
    GpuDisplacementField();
    GpuDisplacementField(
        const stk::GpuVolume& df,
        const AffineTransform& affine_transform = AffineTransform()
    );
    ~GpuDisplacementField();

    // Clones the displacement field and underlying data
    GpuDisplacementField clone();

    // Copies another displacement field into this
    void copy_from(const GpuDisplacementField& other);

    // Returns the dimensions of the displacement field
    dim3 size() const;

    // Volume containing the displacements only
    const stk::GpuVolume& volume() const;
    const AffineTransform& affine_transform() const;

    const float3& origin() const;
    const float3& spacing() const;
    const Matrix3x3f& direction() const;

private:
    stk::GpuVolume _df;
    AffineTransform _affine_transform;
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
    DisplacementField(const GpuDisplacementField& df,
                      const AffineTransform& affine_transform) :
        _df(df.volume()),
        _affine_transform(affine_transform),
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

    // p        : Index in displacement field
    // Returns coordinates in world space
    __device__ float3 transform_index(const int3& p) const
    {
        float4 d = get(p);
        return index2point(p) + float3{d.x, d.y, d.z};
    }

    // Transforms the index with a given delta
    // p        : Index in displacement field
    // delta    : Delta to apply to the transformation 
    // Returns coordinates in world space
    __device__ float3 transform_index(const int3& p, const float4& delta) const
    {
        float4 d = get(p, delta);
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
    AffineTransform _affine_transform;
    dim3 _dims;

    float3 _origin;
    float3 _spacing;
    float3 _inv_spacing;
    Matrix3x3f _direction;
};
} // namespace cuda
#endif // __CUDACC__
