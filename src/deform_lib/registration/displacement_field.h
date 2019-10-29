#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>

#include "settings.h"

class DisplacementField
{
public:
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
                p.z + delta.y / _df.spacing().y,
                p.y + delta.z / _df.spacing().z
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

    // delta : Delta in world space (mm)
    inline void update(const int3& p, const float3& delta)
    {
        _df(p) = get(p, delta);
    }

    // p : Index in displacement field
    // Returns coordinates in world space
    inline float3 transform_index(const int3& p)
    {
        return _df.index2point(p) + get(p);
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

    inline const float3& operator()(int x, int y, int z) const
    {
        return _df(p);
    }
    inline float3& operator()(int x, int y, int z)
    {
        return _df(p);
    }

    inline const float3& operator()(const int3& p) const
    {
        return _df(p);
    }
    inline float3& operator()(const int3& p)
    {
        return _df(p);
    }

private:
    Settings::UpdateRule _update_rule;
    stk::VolumeFloat3 _df;

};