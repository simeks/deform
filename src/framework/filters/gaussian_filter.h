#pragma once

class Volume;

namespace filters
{
    /// Gaussian filter for 3d volumes
    Volume gaussian_filter_3d(const Volume& volume, float sigma);
}