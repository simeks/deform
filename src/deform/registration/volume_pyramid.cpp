#include "volume_pyramid.h"

#include <framework/debug/assert.h>
#include <framework/filters/resample.h>
#include <framework/volume/volume.h>


VolumePyramid::VolumePyramid() : 
    _levels(0), 
    _save_residuals(false)
{
}
VolumePyramid::~VolumePyramid()
{
}
void VolumePyramid::set_level_count(int levels)
{
    _levels = levels;
    _volumes.resize(_levels);
}
void VolumePyramid::build_from_base(const Volume& base, DownsampleFn downsample_fn)
{
    assert(base.valid());
    assert(downsample_fn);
    assert(_levels > 0);

    _save_residuals = false;

    _volumes[0] = base;
    for (int i = 0; i < _levels-1; ++i)
    {
        _volumes[i+1] = downsample_fn(_volumes[i], 0.5f);
    }
}
void VolumePyramid::build_from_base_with_residual(const Volume& base, 
    DownsampleWithResidualFn downsample_fn)
{
    assert(base.valid());
    assert(downsample_fn);
    assert(_levels > 0);

    _save_residuals = true;
    _residuals.resize(_levels);

    _volumes[0] = base;
    for (int i = 0; i < _levels-1; ++i)
    {
        _volumes[i+1] = downsample_fn(_volumes[i], 0.5f, _residuals[i]);
    }
}
void VolumePyramid::set_volume(int level, const Volume& vol)
{
    assert(level < _levels);
    _volumes[level] = vol;
}
const Volume& VolumePyramid::volume(int level) const
{
    assert(level < _levels);
    return _volumes[level];
}
const Volume& VolumePyramid::residual(int level) const
{
    assert(_save_residuals);
    assert(level < _levels);
    return _residuals[level];
}
int VolumePyramid::levels() const
{
    return _levels;
}
