#pragma once

#include "config.h"

#include <framework/volume/volume_helper.h>

#ifdef DF_ENABLE_VOXEL_CONSTRAINTS
namespace hard_constraints
{
    VolumeUInt8 downsample_mask_by_2(const VolumeUInt8& mask);
    VolumeFloat3 downsample_values_by_2(const VolumeUInt8& mask, const VolumeFloat3& values);
}
#endif // DF_ENABLE_VOXEL_CONSTRAINTS
