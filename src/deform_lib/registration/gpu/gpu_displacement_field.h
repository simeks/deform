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

namespace cuda {

#ifdef __CUDACC__

struct CompositiveUpdate
{
    __device__ float3 operator()(
        const stk::cuda::VolumePtr<float4>& df,
        const dim3& dims,
        const float3& inv_spacing,
        const int3& index,
        const float4& delta
    ) {
        // Convert delta from mm to image space
        float4 o = stk::cuda::linear_at_clamp<float4>(
            df,
            dims,
            index.x + delta.x * inv_spacing.x,
            index.y + delta.y * inv_spacing.y,
            index.z + delta.z * inv_spacing.z
        ) + delta;
        return float3{o.x, o.y, o.z};
    }
};

struct AdditiveUpdate
{
    __device__ float3 operator()(
        const stk::cuda::VolumePtr<float4>& df,
        const dim3& /*dims*/,
        const float3& /*inv_spacing*/,
        const int3& index,
        const float4& delta
    ) {
        float4 o = df(index.x, index.y, index.z) + delta;
        return float3{o.x, o.y, o.z};
    }
};


template<typename TUpdateFn = AdditiveUpdate>
class DisplacementField
{
public:
    DisplacementField(const GpuDisplacementField& df) :
        _df(df.volume()),
        _affine(df.affine_transform()),
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
        float3 out = transform_index(p) - index2point(p);
        return float4{out.x, out.y, out.z, 0};
    }

    __device__ float4 get(const int3& p, const float4& delta) const
    {
        float3 p1 = index2point(p);
        
        TUpdateFn fn;
        float3 p2 = p1 + fn(_df, _dims, _inv_spacing, p, delta);
        float3 p3 = _affine.transform_point(p2);

        float3 out = p3 - p1;
        return float4{out.x, out.y, out.z, 0};
    }

    // p        : Index in displacement field
    // Returns coordinates in world space
    __device__ float3 transform_index(const int3& p) const
    {
        float4 d = _df(p.x, p.y, p.z);
        return _affine.transform_point(
            index2point(p) + float3{d.x, d.y, d.z}
        );
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
    ::AffineTransform _affine;
    dim3 _dims;

    float3 _origin;
    float3 _spacing;
    float3 _inv_spacing;
    Matrix3x3f _direction;
    Matrix3x3f _inv_direction;
};

#endif // __CUDACC__


// Computes the composite of a vector field and an affine transform:
// u'(x) <- A(x + u(x)) + b - x
stk::GpuVolume compute_displacement_field(
    const stk::GpuVolume& vector_field,
    const AffineTransform& affine
);

} // namespace cuda
