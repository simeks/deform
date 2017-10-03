#pragma once

class Volume;
struct Dims;

namespace filters
{
    /// Downsamples a mask volume
    /// Each subvoxel takes the max value from the corresponding super voxels
    Volume downsample_mask(const Volume& vol, float scale);

    /// Applies a gaussian prefilter and downsamples the volume by the given scale factor.
    Volume downsample_volume_gaussian(const Volume& vol, float scale);

    /// Downsamples a 3d vectorfield and keeps the residual
    Volume downsample_vectorfield(const Volume& vol, float scale, Volume& residual);

    /// Upsamples a 3d vectorfield and optionally applies the given residual
    /// residual : Passing an invalid Volume (simply Volume()) will behave the same as having a zero residual
    Volume upsample_vectorfield(const Volume& vol, const Dims& new_dims, const Volume& residual);
}