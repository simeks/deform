#pragma once

namespace stk
{
    class Volume;
}

struct dim3;

namespace filters
{
    /// Applies a gaussian prefilter and downsamples the volume by the given scale factor.
    stk::Volume downsample_volume_gaussian(const stk::Volume& vol, float scale);

    /// Downsamples a 3d vectorfield and keeps the residual
    stk::Volume downsample_vectorfield(const stk::Volume& vol, float scale, stk::Volume& residual);

    /// Upsamples a 3d vectorfield and optionally applies the given residual
    /// residual : Passing an invalid Volume (simply Volume()) will behave the same as having a zero residual
    stk::Volume upsample_vectorfield(const stk::Volume& vol, const dim3& new_dims, const stk::Volume& residual);
}