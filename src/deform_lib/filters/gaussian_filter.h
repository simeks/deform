#pragma once

namespace stk
{
    class Volume;
}

namespace filters
{
    /// Gaussian filter for 3d volumes
    /// Performs per-component filtering for multi-channel volumes.
    stk::Volume gaussian_filter_3d(const stk::Volume& volume, float sigma);
}