#include "volume_pyramid.h"

#include "../filters/resample.h"

#include <stk/common/assert.h>
#include <stk/image/volume.h>


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
void VolumePyramid::build_from_base(const stk::Volume& base, 
    DownsampleFn downsample_fn)
{
    ASSERT(base.valid());
    ASSERT(downsample_fn);
    ASSERT(_levels > 0);

    _save_residuals = false;

    _volumes[0] = base;
    for (int i = 0; i < _levels-1; ++i) {
        _volumes[i+1] = downsample_fn(_volumes[i], 0.5f);
    }
}
void VolumePyramid::build_from_base_with_residual(const stk::Volume& base, 
    DownsampleWithResidualFn downsample_fn)
{
    ASSERT(base.valid());
    ASSERT(downsample_fn);
    ASSERT(_levels > 0);

    _save_residuals = true;
    _residuals.resize(_levels);

    _volumes[0] = base;
    for (int i = 0; i < _levels-1; ++i) {
        _volumes[i+1] = downsample_fn(_volumes[i], 0.5f, _residuals[i]);
    }
}
void VolumePyramid::set_volume(int level, const stk::Volume& vol)
{
    ASSERT(level < _levels);
    _volumes[level] = vol;
}
const stk::Volume& VolumePyramid::volume(int level) const
{
    ASSERT(level < _levels);
    return _volumes[level];
}
const stk::Volume& VolumePyramid::residual(int level) const
{
    ASSERT(_save_residuals);
    ASSERT(level < _levels);
    return _residuals[level];
}
int VolumePyramid::levels() const
{
    return _levels;
}
