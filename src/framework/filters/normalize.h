#pragma once

#include <framework/volume/volume_helper.h>

namespace filters
{
    /// Normalizes a scalar volume to the specified range
    template<typename T>
    VolumeHelper<T> normalize(const VolumeHelper<T>& in, T min, T max);
}

#include "normalize.inl"