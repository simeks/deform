#pragma once

#include <stk/image/volume.h>

class DisplacementField;

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
    const DisplacementField& df,
    transform::Interp i = transform::Interp_Linear
);

stk::Volume transform_volume(
    const stk::Volume& src,
    const stk::VolumeFloat3& df,
    transform::Interp i = transform::Interp_Linear
);
