#pragma once

#include <stk/image/volume.h>

#define JAC_TYPE double

stk::Volume calculate_jacobian(const stk::VolumeFloat3& def);
