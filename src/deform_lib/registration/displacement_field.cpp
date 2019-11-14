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
            // Transform consists of two components, affine and displacement vector
            //  V(x) = Ax + b
            //  W(x) = x + u(x)
            //
            // The composition is computed as
            //  V(W(x)) = A(x + u(x)) + b
            //
            // Applying delta, produces following eq
            // V(W'(x)) = A(x + u'(x)) + b = A(x + u(x + delta) + delta) + b
            //
            // Applying the delta only requires us to modify u(x)
            // u'(x) = u(x + delta) + delta

            float3 delta = update_field.get(p);
            float3 p1 = _df.index2point(p);
            _df(p) = _df.linear_at_point(p1 + delta, stk::Border_Replicate) + delta;
        }
        else {
            // For additive updates we simply add the delta to the displacement

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

stk::VolumeFloat3 compute_displacement_field(
    const stk::VolumeFloat3& vector_field,
    const AffineTransform& affine
)
{
    dim3 dims = vector_field.size();
    stk::VolumeFloat3 out(dims);
    out.copy_meta_from(vector_field);

    DisplacementField df(vector_field, affine);

    #pragma omp parallel for
    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        out(x,y,z) = df.get(int3{x, y, z});
    }}}

    return out;
}
