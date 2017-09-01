#pragma once

/// Image pyramid representing a set of images each smaller than the other.
/// I.e. 
/// Level:                      Size [elements]:
///     2       | | |           2
///     1     | | | | |         4
///     0 | | | | | | | | |     8
///
/// Level 0 is our base image, typically the images used as input for the registration (fixed or moving)
/// For each step the size of the image is halved
class VolumePyramid
{
public:
    typedef Volume (*ResampleVolumeFn)(const Volume&, float scale);

    VolumePyramid(int levels);
    ~VolumePyramid();

    /// Sets the given base at index 0 and builds the rest of the pyramid using
    /// the provided resampling function.
    void build_from_base(const Volume& base, ResampleVolumeFn resample_fn);

private:
    ResampleVolumeFn _resample_fn;

    Volume* _volumes;
};
