#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>

#include "settings.h"

/** Wrapper for a displacement field image.
 * 
 * Users are adviced to use the transform_* methods, rather then applying the
 * displacements themselves via `df.get(p)`.
 * */
class DisplacementField
{
public:
    DisplacementField() : _update_rule(Settings::UpdateRule_Additive) {}
    DisplacementField(
        const stk::VolumeFloat3& df,
        Settings::UpdateRule update_rule = Settings::UpdateRule_Additive
    ) :
        _update_rule(update_rule),
        _df(df)
    {
    }
    ~DisplacementField() {}

    inline float3 get(const int3& p) const
    {
        return _df(p);
    }

    // delta : Delta in world space (mm)
    inline float3 get(const int3& p, const float3& delta) const
    {
        if (_update_rule == Settings::UpdateRule_Compositive) {
            // Convert delta to fixed image space
            // TODO: What about orientation?
            float3 fp {
                p.x + delta.x / _df.spacing().x,
                p.y + delta.y / _df.spacing().y,
                p.z + delta.z / _df.spacing().z
            };
            return _df.linear_at(fp, stk::Border_Replicate) + delta;
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
        return _df.index2point(p) + get(p);
    }

    dim3 size() const
    {
        return _df.size();
    }

    DisplacementField clone() const
    {
        return DisplacementField(_df.clone(), _update_rule);
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

    Settings::UpdateRule update_rule() const
    {
        return _update_rule;
    }

private:
    Settings::UpdateRule _update_rule;
    stk::VolumeFloat3 _df;

};