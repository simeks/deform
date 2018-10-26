#pragma once

#include "../config.h"

#include <vector>

namespace stk
{
    class GpuVolume;
}

/// GPU version of VolumePyramid
/// @sa VolumePyramid
class GpuVolumePyramid
{
public:
    typedef stk::GpuVolume (*DownsampleFn)(const stk::GpuVolume&);

    GpuVolumePyramid();
    ~GpuVolumePyramid();

    /// Sets the size of the pyramid.
    /// This needs to be called before build_from_base, otherwise you'll need to rebuild the pyramid
    void set_level_count(int levels);

    /// Sets the given base at index 0 and builds the rest of the pyramid using
    /// the provided resampling function.
    /// downsample_fn : Resampling function, required to support downsampling
    void build_from_base(const stk::GpuVolume& base, DownsampleFn downsample_fn);

    /// Sets the volume at the given level without rebuilding any other levels of the pyramid.
    void set_volume(int level, const stk::GpuVolume& vol);

    /// Returns the volume at the specified level, 0 being the base volume.
    const stk::GpuVolume& volume(int level) const;

    /// Returns the number of levels in this pyramid
    int levels() const;

private:
    int _levels;

    std::vector<stk::GpuVolume> _volumes;
};
