#include "volume_pyramid.h"

#include "../filters/resample.h"

#include <stk/common/assert.h>
#include <stk/image/volume.h>


VolumePyramid::VolumePyramid() : 
    _levels(0)
#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    , _save_residuals(false)
#endif
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

#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    _save_residuals = false;
#endif

    _volumes[0] = base;
    for (int i = 0; i < _levels-1; ++i) {
        _volumes[i+1] = downsample_fn(_volumes[i]);
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
int VolumePyramid::levels() const
{
    return _levels;
}
#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
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
        _volumes[i+1] = downsample_fn(_volumes[i], _residuals[i]);
    }
}
const stk::Volume& VolumePyramid::residual(int level) const
{
    ASSERT(_save_residuals);
    ASSERT(level < _levels);
    return _residuals[level];
}
#endif // DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
