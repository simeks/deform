#pragma once

#include <stk/image/volume.h>

namespace transform
{
    enum Interp : uint8_t
    {
        Interp_NN,
        Interp_Linear
    };
}

stk::Volume transform_volume(
    const stk::Volume& src,
    const stk::VolumeFloat3& def,
    transform::Interp i = transform::Interp_Linear
);
