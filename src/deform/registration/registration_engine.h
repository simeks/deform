#pragma once

class ConfigFile;
class Optimizer;
class Volume;

class RegistrationEngine
{
public:

    RegistrationEngine();
    ~RegistrationEngine();

    bool initialize(const ConfigFile& cfg);
    void shutdown();

   void set_image_pair(int i, const Volume& fixed, const Volume& moving);


private:
    void build_pyramid();

    // uint32_t _pyramid_levels;

    // ResampleVolumeFn* _resampler;
    // Volume** _fixed_pyramid;
    // Volume** _moving_pyramid;
    // VolumeVec3f** _deformation_pyramid;

    Optimizer* optimizer;
};