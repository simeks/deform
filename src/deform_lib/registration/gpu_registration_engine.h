#pragma once

#include "../config.h"
#include "settings.h"
#include "gpu_volume_pyramid.h"

#include <stk/image/volume.h>

#include <vector>

class GpuRegistrationEngine
{
public:
    GpuRegistrationEngine(const Settings& settings);
    ~GpuRegistrationEngine();

    void set_initial_deformation(const stk::GpuVolume& def);
    void set_image_pair(
        int i, 
        const stk::GpuVolume& fixed, 
        const stk::GpuVolume& moving,
        stk::GpuVolume (*downsample_fn)(const stk::GpuVolume&));

    /// Runs the registration. 
    /// Returns the resulting deformation field or an invalid volume if registration failed.
    stk::GpuVolume execute();
    
private:
    Settings _settings;

    std::vector<GpuVolumePyramid> _fixed_pyramids;
    std::vector<GpuVolumePyramid> _moving_pyramids;
    GpuVolumePyramid _deformation_pyramid;

};
