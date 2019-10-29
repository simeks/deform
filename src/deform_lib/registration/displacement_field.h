#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>

#include "settings.h"

float3 CompositiveUpdate(
    const stk::VolumeFloat3& df,
    const int3& p,
    const float3& delta
) {
    // Convert delta to fixed image space
    // TODO: What about orientation?
    float3 fp {
        p.x + delta.x / df.spacing().x,
        p.z + delta.y / df.spacing().y,
        p.y + delta.z / df.spacing().z
    };
    return df.linear_at(fp, stk::Border_Replicate) + delta;
}

float3 AdditiveUpdate(
    const stk::VolumeFloat3& df,
    const int3& p,
    const float3& delta
) {
    return df(p) + delta;
}

typedef float3 (*UpdateFn)(const stk::VolumeFloat3&, const int3&, const float3&);

class DisplacementField
{
public:
    DisplacementField(
        Settings::UpdateRule update_rule,
        const stk::VolumeFloat3& df
    ) : _df(df)
    {
        if (update_rule == Settings::UpdateRule_Compositive)
            _update_fn = CompositiveUpdate;
        else
            _update_fn = AdditiveUpdate;
        
    }
    ~DisplacementField() {}

    inline float3 get(const int3& p)
    {
        return _df(p);
    }

    // delta : Delta in world space (mm)
    inline float3 get(const int3& p, const float3& delta)
    {
        return _update_fn(_df, p, delta);
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

    dim3 size() const
    {
        return _df.size();
    }

    // Volume containing the displacements only
    const stk::VolumeFloat3& volume() const
    {
        return _df;
    }

private:
    UpdateFn _update_fn;
    stk::VolumeFloat3 _df;

};