#pragma once

#include <stk/image/volume.h>

stk::Volume calculate_jacobian(
    const stk::Volume& src, 
    const stk::VolumeFloat3& def
);
