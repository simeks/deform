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

    void set_initial_deformation(const stk::Volume& def);
    void set_image_pair(
        int i, 
        const stk::Volume& fixed, 
        const stk::Volume& moving
    );

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_regularization_weight_map(const stk::Volume& map);
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    /// Sets fixed and moving landmarks.
    void set_landmarks(const std::vector<float3>& fixed_landmarks,
                       const std::vector<float3>& moving_landmarks);

    /// Sets mask and values for constraints
    void set_voxel_constraints(const stk::VolumeUChar& mask, const stk::VolumeFloat3& values);

    /// Runs the registration. 
    /// Returns the resulting deformation field or an invalid volume if registration failed.
    stk::Volume execute();
    
private:
    Settings _settings;

    std::vector<GpuVolumePyramid> _fixed_pyramids;
    std::vector<GpuVolumePyramid> _moving_pyramids;
    GpuVolumePyramid _deformation_pyramid;

};
