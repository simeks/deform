#pragma once

#include "../config.h"

#include <stk/image/volume.h>

class VolumePyramid;

namespace voxel_constraints
{
    stk::VolumeUChar downsample_mask_by_2(const stk::VolumeUChar& mask);
    stk::VolumeFloat3 downsample_values_by_2(const stk::VolumeUChar& mask, 
        const stk::VolumeFloat3& values);

    void build_pyramids(const stk::VolumeUChar& mask, const stk::VolumeFloat3& values,
        int num_levels, VolumePyramid& mask_pyramid, VolumePyramid& values_pyramid);
}
