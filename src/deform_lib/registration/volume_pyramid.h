#pragma once

#include "../config.h"

#include <vector>

namespace stk
{
    class Volume;
}

/// Image pyramid representing a set of images each smaller than the other.
/// I.e.
/// Level:                      Size [elements]:
///     3        | |            1
///     2       | | |           2
///     1     | | | | |         4
///     0 | | | | | | | | |     8
///
/// Level 0 is our base image, typically the images used as input for the registration (fixed or moving)
/// For each step the size of the image is halved
class VolumePyramid
{
public:
    typedef stk::Volume (*DownsampleFn)(const stk::Volume&);

#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    typedef stk::Volume (*DownsampleWithResidualFn)(const stk::Volume&, stk::Volume& residual);
#endif

    VolumePyramid();
    ~VolumePyramid();

    /// Sets the size of the pyramid.
    /// This needs to be called before build_from_base, otherwise you'll need to rebuild the pyramid
    void set_level_count(int levels);

    /// Sets the given base at index 0 and builds the rest of the pyramid using
    /// the provided resampling function.
    /// downsample_fn : Resampling function, required to support downsampling
    void build_from_base(const stk::Volume& base, DownsampleFn downsample_fn);

#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    /// Sets the given base at index 0 and builds the rest of the pyramid using
    /// the provided resampling function.
    /// Using this method in comparison to build_from_base allows for saving the residuals of the
    ///     downsampling. This could be useful for deformation fields.
    /// downsample_fn : Resampling function, required to support downsampling
    void build_from_base_with_residual(const stk::Volume& base, DownsampleWithResidualFn downsample_fn);
#endif

    /// Sets the volume at the given level without rebuilding any other levels of the pyramid.
    void set_volume(int level, const stk::Volume& vol);

    /// Returns the volume at the specified level, 0 being the base volume.
    const stk::Volume& volume(int level) const;

#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    /// Returns the residual volume at the specified level, 0 being the base volume
    /// This will fail if the pyramid wasn't built using build_*_with_residual
    const stk::Volume& residual(int level) const;
#endif

    /// Returns the number of levels in this pyramid
    int levels() const;

private:
    int _levels;

    std::vector<stk::Volume> _volumes;

#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    bool _save_residuals;

    /// Residuals:
    /// Index 0 holds the residual from downsampling pyramid level 0 to 1, index 1 holds 1 to 2, ...
    std::vector<stk::Volume> _residuals;
#endif
};
