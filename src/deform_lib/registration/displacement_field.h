#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>

struct CompositiveUpdate
{
    inline float3 operator()(
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
};

struct AdditiveUpdate
{
    inline float3 operator()(
        const stk::VolumeFloat3& df,
        const int3& p,
        const float3& delta
    ) {
        return df(p) + delta;
    }
};

template<typename TUpdate>
class DisplacementField
{
public:
    DisplacementField(const stk::VolumeFloat3& df) : _df(df) {}
    ~DisplacementField() {}

    inline float3 get(const int3& p)
    {
        return _df(p);
    }

    // delta : Delta in world space (mm)
    inline float3 get(const int3& p, const float3& delta)
    {
        TUpdate _update_fn;
        return _update_fn(_df, p, delta);
    }

    inline void set(const int3& p, const float3& d)
    {
        _df(p) = d;
    }

    // delta : Delta in world space (mm)
    inline void update(const int3& p, const float3& delta)
    {
        TUpdate _update_fn;
        _df(p) = _update_fn(_df, p, delta);
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
    stk::VolumeFloat3 _df;

};