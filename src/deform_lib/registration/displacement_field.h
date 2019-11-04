#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>

#include "affine.h"
#include "settings.h"

/** Wrapper for a displacement field image.
 * 
 * Users are adviced to use the transform_* methods, rather then applying the
 * displacements themselves via `df.get(p)`.
 * */
class DisplacementField
{
public:
    DisplacementField() {}
    DisplacementField(const stk::VolumeFloat3& df) :
        _df(df)
    {
    }
    DisplacementField(const dim3& dims) :
        _df(dims, float3{0, 0, 0})
    {
    }
    ~DisplacementField() {}

    inline float3 get(const int3& p) const
    {
        return _df(p);
    }

    // delta : Delta in world space (mm)
    inline float3 get(const int3& p, const float3& delta, bool composite) const
    {
        if (composite) {
            float3 p1 = _df.index2point(p);
            float3 p2 = p1 + delta;
            float3 p3 = p2 + _df.linear_at_point(p2, stk::Border_Replicate);

            return p3 - p1;
        }
        else /*(_update_rule == Settings::UpdateRule_Additive)*/ {
            return _df(p) + delta;
        }
    }

    inline void set(const int3& p, const float3& d)
    {
        _df(p) = d;
    }

    // p : Index in displacement field
    // Returns coordinates in world space
    inline float3 transform_index(const int3& p) const
    {
        return _df.index2point(p) + _df(p);
    }

    void update(const DisplacementField& update_field, bool composite)
    {
        for (int3 p : _df.size()) {
            if (composite) {
                float3 p1 = _df.index2point(p);
                float3 p2 = p1 + update_field.get(p);
                float3 p3 = p2 + _df.linear_at_point(p2, stk::Border_Replicate);

                _df(p) = p3 - p1;
            }
            else {
                _df(p) += update_field.get(p);
            }
        }
    }

    void set_affine_transform(const AffineTransform& transform)
    {
        _affine = transform;
    }

    dim3 size() const
    {
        return _df.size();
    }

    void copy_from(const DisplacementField& other)
    {
        _df.copy_from(other._df);
    }

    // Volume containing the displacements only
    const stk::VolumeFloat3& volume() const
    {
        return _df;
    }

    // Volume containing the displacements only
    stk::VolumeFloat3& volume()
    {
        return _df;
    }

    // Returns true if the volume is allocated and ready for use
    bool valid() const
    {
        return _df.valid();
    }


private:
    stk::VolumeFloat3 _df;

    AffineTransform _affine;

};
