#include "gpu_volume_pyramid.h"

#include <stk/common/assert.h>
#include <stk/image/gpu_volume.h>


GpuVolumePyramid::GpuVolumePyramid() :
    _levels(0)
{
}
GpuVolumePyramid::~GpuVolumePyramid()
{
}
void GpuVolumePyramid::set_level_count(int levels)
{
    _levels = levels;
    _volumes.resize(_levels);
}
void GpuVolumePyramid::build_from_base(const stk::GpuVolume& base,
    DownsampleFn downsample_fn)
{
    ASSERT(base.valid());
    ASSERT(downsample_fn);
    ASSERT(_levels > 0);

    _volumes[0] = base;
    for (int i = 0; i < _levels-1; ++i) {
        _volumes[i+1] = downsample_fn(_volumes[i]);
    }
}
void GpuVolumePyramid::set_volume(int level, const stk::GpuVolume& vol)
{
    ASSERT(level < _levels);
    _volumes[level] = vol;
}
const stk::GpuVolume& GpuVolumePyramid::volume(int level) const
{
    ASSERT(level < _levels);
    return _volumes[level];
}
int GpuVolumePyramid::levels() const
{
    return _levels;
}
