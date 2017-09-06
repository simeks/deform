#pragma once

#include "volume_pyramid.h"

#include <vector>

class ConfigFile;
class Optimizer;

struct ImagePairDesc
{
    Volume (*downsample_fn)(const Volume&, float);
};

class RegistrationEngine
{
public:
    enum ImageType
    {
        Image_Float,
        Image_Mask
    };

    RegistrationEngine(int pyramid_levels, int image_pair_count);
    ~RegistrationEngine();

    void set_initial_deformation(const Volume& def);
    void set_image_pair(int i, const ImagePairDesc& desc, const Volume& fixed, const Volume& moving);

    /// Runs the engine
    /// Returns the resulting deformation field or an invalid volume if registration failed.
    Volume run();

private:
    void build_pyramid();

    int _pyramid_levels; // Size of the multi-res pyramids
    int _image_pair_count; // Number of image pairs (e.g. fat, water and mask makes 3)

    std::vector<VolumePyramid> _fixed_pyramids;
    std::vector<VolumePyramid> _moving_pyramids;
    VolumePyramid _deformation_pyramid;

    
    Optimizer* optimizer;
};