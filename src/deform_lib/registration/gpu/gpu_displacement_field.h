#pragma once
#pragma once

#include <stk/image/gpu_volume.h>
#include <stk/math/float3.h>

#include "settings.h"

/** Wrapper for a displacement field image.
 * 
 * Users are adviced to use the transform_* methods, rather then applying the
 * displacements themselves via `df.get(p)`.
 * */
class GpuDisplacementField
{
public:
    GpuDisplacementField(
        const stk::GpuVolume& df,
        Settings::UpdateRule update_rule = Settings::UpdateRule_Additive
    ) :
        _update_rule(update_rule),
        _df(df)
    {
    }
    ~GpuDisplacementField() {}

    GpuDisplacementField clone()
    {
        return GpuDisplacementField(_df.clone(), _update_rule);
    }

    void copy_from(const GpuDisplacementField& other)
    {
        _df.copy_from(other.volume());
    }

    dim3 size() const
    {
        return _df.size();
    }

    // Volume containing the displacements only
    const stk::GpuVolume& volume() const
    {
        return _df;
    }

private:
    Settings::UpdateRule _update_rule;
    stk::GpuVolume _df;

};