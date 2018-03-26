#pragma once

#include <framework/volume/volume_helper.h>

namespace transform
{
    enum Interp : uint8_t
    {
        Interp_NN,
        Interp_Linear
    };
}

Volume transform_volume(
    const Volume& src, 
    const VolumeFloat3& def, 
    transform::Interp i = transform::Interp_Linear
);
