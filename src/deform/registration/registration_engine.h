#pragma once

#include "config.h"
#include "settings.h"
#include "volume_pyramid.h"

#ifdef DF_ENABLE_VOXEL_CONSTRAINTS
#include <framework/volume/volume_helper.h>
#endif // DF_ENABLE_VOXEL_CONSTRAINTS

#include <vector>

class RegistrationEngine
{
public:
    RegistrationEngine(const Settings& settings);
    ~RegistrationEngine();

    void initialize(int image_pair_count);

    void set_initial_deformation(const Volume& def);
    void set_image_pair(
        int i, 
        const Volume& fixed, 
        const Volume& moving,
        Volume (*downsample_fn)(const Volume&, float));

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_regularization_weight_map(const Volume& map);
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

#ifdef DF_ENABLE_VOXEL_CONSTRAINTS
    /// Sets mask and values for constraints
    void set_voxel_constraints(const VolumeUInt8& mask, const VolumeFloat3& values);
#endif // DF_ENABLE_VOXEL_CONSTRAINTS

    /// Runs the registration. 
    /// Returns the resulting deformation field or an invalid volume if registration failed.
    Volume execute();

    /// Validates all volumes and makes sure everything is in order.
    /// Should be called before performing executing the registration.
    /// Returns true if the validation was successful, false if not.
    bool validate_input();

private:
#ifdef DF_OUTPUT_DEBUG_VOLUMES
    /// Upsamples the deformation field at the given level and saves it
    void upsample_and_save(int level);
    
    /// Saves the complete volume pyramids
    void save_volume_pyramid();
#endif
    
    Settings _settings;
    int _image_pair_count; // Number of image pairs (e.g. fat, water and mask makes 3)

    std::vector<VolumePyramid> _fixed_pyramids;
    std::vector<VolumePyramid> _moving_pyramids;
    VolumePyramid _deformation_pyramid;

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    VolumePyramid _regularization_weight_map;
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP


#ifdef DF_ENABLE_VOXEL_CONSTRAINTS
    VolumePyramid _constraints_pyramid;
    VolumePyramid _constraints_mask_pyramid;
#endif // DF_ENABLE_VOXEL_CONSTRAINTS

};
