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
    DisplacementField(
        const stk::VolumeFloat3& df,
        const AffineTransform& affine = AffineTransform()
    ) :
        _df(df),
        _affine(affine)
    {
    }
    DisplacementField(const dim3& dims) :
        _df(dims, float3{0, 0, 0})
    {
    }
    ~DisplacementField() {}

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
        else /*(_update_rule == Settings::UpdateRule_Additive)*/ {
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


        float3 d1 = p + _df.linear_at_point(p, stk::Border_Replicate);
        
        float3 d2 = _affine.transform_point(
            p + _df.linear_at_point(p, stk::Border_Replicate)
        );

        // if (abs(d1.x - d2.x) > 0.000000001f
        //  || abs(d1.y - d2.y) > 0.000000001f
        //  || abs(d1.z - d2.z) > 0.000000001f) {
        //     LOG(Info) << "pt: " << d1 << " != " << d2;
        //     LOG(Info) << p;
        // }

        return d2;
    }

    // p : Index in displacement field
    // Returns coordinates in world space
    inline float3 transform_index(const int3& p) const
    {

        float3 d1 = _df.index2point(p) + _df(p);
        float3 d2 = _affine.transform_point(_df.index2point(p) + _df(p));
        
        return d2;
    }

    void update(const DisplacementField& update_field, bool composite)
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

    void fill(const float3& v)
    {
        _df.fill(v);
    }

    void set_affine_transform(const AffineTransform& transform)
    {
        _affine = transform;
    }

    DisplacementField clone() const
    {
        return DisplacementField(_df.clone(), _affine);
    }

    dim3 size() const
    {
        return _df.size();
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
