#include "displacement_field.h"



DisplacementField::DisplacementField() {}
DisplacementField::DisplacementField(
    const stk::VolumeFloat3& df,
    const AffineTransform& affine
) :
    _df(df),
    _affine(affine)
{
}
DisplacementField::DisplacementField(const dim3& dims) :
    _df(dims, float3{0, 0, 0})
{
}
DisplacementField::~DisplacementField() {}

void DisplacementField::update(const DisplacementField& update_field, bool composite)
{
    dim3 dims = update_field.size();
    
    DisplacementField buffer = this->clone();

    #pragma omp parallel for
    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        int3 p {x, y, z};
        if (composite) {
            float3 p1 = _df.index2point(p);
            float3 p2 = p1 + update_field.get(p);
            float3 p3 = buffer.transform_point(p2);

            _df(p) = p3 - p1;
        }
        else {
            _df(p) += update_field.get(p);
        }
    }}}
}
void DisplacementField::fill(const float3& v)
{
    _df.fill(v);
}
DisplacementField DisplacementField::clone() const
{
    return DisplacementField(_df.clone(), _affine);
}
dim3 DisplacementField::size() const
{
    return _df.size();
}
const stk::VolumeFloat3& DisplacementField::volume() const
{
    return _df;
}
stk::VolumeFloat3& DisplacementField::volume()
{
    return _df;
}
bool DisplacementField::valid() const
{
    return _df.valid();
}

