#include "volume_pyramid.h"

#include <framework/debug/assert.h>
#include <framework/volume/volume.h>

VolumePyramid::VolumePyramid(int levels) : _levels(levels)
{
    _volumes.resize(_levels);
}
VolumePyramid::~VolumePyramid()
{
}

void VolumePyramid::build_from_base(const Volume& base, ResampleVolumeFn resample_fn)
{
    assert(resample_fn);
    assert(_levels > 0);

    _volumes[0] = base;
    for (int i = 0; i < _levels-1; ++i)
    {
        _volumes[i+1] = resample_fn(_volumes[i], 0.5f);
    }
}
