#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>

#include "affine_transform.h"
#include "settings.h"

/** Wrapper for a displacement field image.
 * 
 * Users are adviced to use the transform_* methods, rather then applying the
 * displacements themselves via `df.get(p)`.
 * */
class DisplacementField
{
public:
    DisplacementField();
    DisplacementField(
        const stk::VolumeFloat3& df,
        const AffineTransform& affine = AffineTransform()
    );
    // Creates a new identity displacement field of size dims
    DisplacementField(const dim3& dims);
    ~DisplacementField();

    // Returns displacement at index p
    // p : Index in displacement field
    inline float3 get(const int3& p) const
    {
        return transform_index(p) - _df.index2point(p);
    }

    // Returns displacement at point p
    // p : Point in world space
    inline float3 get(const float3& p) const
    {
        return transform_point(p) - p;
    }

    // delta : Delta in world space (mm)
    inline float3 get(const int3& p, const float3& delta, bool composite) const
    {
        if (composite) {
            float3 p1 = _df.index2point(p);
            float3 p2 = p1 + delta;
            float3 p3 = transform_point(p2);

            return p3 - p1;
        }
        else {
            return get(p) + delta;
        }
    }

    // Sets the displacement at given index p to value d.
    // This modifies the displacement field directly with no regards to the affine
    // transformation
    inline void set(const int3& p, const float3& d)
    {
        _df(p) = d;
    }

    // p : Point in world space
    // Returns coordinates in world space
    inline float3 transform_point(const float3& p) const
    {
        return _affine.transform_point(
            p + _df.linear_at_point(p, stk::Border_Replicate)
        );
    }

    // p : Index in displacement field
    // Returns coordinates in world space
    inline float3 transform_index(const int3& p) const
    {
        return _affine.transform_point(_df.index2point(p) + _df(p));
    }

    void update(const DisplacementField& update_field, bool composite);

    // Fills the displacement field component with vector v
    void fill(const float3& v);

    // Clones the displacement field and any affine transformation
    DisplacementField clone() const;

    // Size of the displacement field
    dim3 size() const;

    // Volume containing the displacements only
    const stk::VolumeFloat3& volume() const;

    // Volume containing the displacements only
    stk::VolumeFloat3& volume();

    // Returns true if the volume is allocated and ready for use
    bool valid() const;

private:
    stk::VolumeFloat3 _df;
    AffineTransform _affine;

};

// Computes the composite of a vector field and an affine transform:
// u'(x) <- A(x + u(x)) + b - x
stk::VolumeFloat3 compute_displacement_field(
    const stk::VolumeFloat3& vector_field,
    const AffineTransform& affine
);
