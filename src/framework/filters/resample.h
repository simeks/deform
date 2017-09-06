#pragma once

class Volume;

namespace filters
{
    /// Applies a gaussian prefilter and downsamples the volume by the given scale factor.
    Volume downsample_volume_gaussian(const Volume& vol, float scale);

    /// Downsamples a 3d vectorfield and keeps the residual
    Volume downsample_vectorfield(const Volume& vol, float scale, Volume& residual);
}