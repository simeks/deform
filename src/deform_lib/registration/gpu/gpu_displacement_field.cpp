#include "gpu_displacement_field.h"

GpuDisplacementField::GpuDisplacementField()
{
}
GpuDisplacementField::GpuDisplacementField(
    const stk::GpuVolume& df,
    const AffineTransform& affine_transform
) :
    _df(df), _affine_transform(affine_transform)
{
    ASSERT(_df.voxel_type() == stk::Type_Float4);
}
GpuDisplacementField::~GpuDisplacementField()
{
}
GpuDisplacementField GpuDisplacementField::clone()
{
    return GpuDisplacementField(_df.clone(), _affine_transform);
}
void GpuDisplacementField::copy_from(const GpuDisplacementField& other)
{
    _df.copy_from(other._df);
    _affine_transform = other._affine_transform;
}
dim3 GpuDisplacementField::size() const
{
    return _df.size();
}
const stk::GpuVolume& GpuDisplacementField::volume() const
{
    return _df;
}
const AffineTransform& GpuDisplacementField::affine_transform() const
{
    return _affine_transform;
}
const float3& GpuDisplacementField::origin() const
{
    return _df.origin();
}
const float3& GpuDisplacementField::spacing() const
{
    return _df.spacing();
}
const Matrix3x3f& GpuDisplacementField::direction() const
{
    return _df.direction();
}
