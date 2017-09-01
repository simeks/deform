#include "volume_pyramid.h"


VolumePyramid::VolumePyramid(int levels);
VolumePyramid::~VolumePyramid();

void VolumePyramid::build_from_base(const Volume& base, ResampleVolumeFn resample_fn)
{
    base; resample_fn;
}
