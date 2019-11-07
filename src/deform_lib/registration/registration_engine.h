#pragma once

#include "../config.h"
#include "settings.h"
#include "volume_pyramid.h"

#include <stk/image/volume.h>

#include <vector>

struct Regularizer;
struct UnaryFunction;

class RegistrationEngine
{
public:
    RegistrationEngine(const Settings& settings);
    ~RegistrationEngine();

    void set_initial_deformation(const stk::Volume& def);
    void set_affine_transform(const AffineTransform& affine_transform);

    void set_image_pair(
        int i,
        const stk::Volume& fixed,
        const stk::Volume& moving);

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_regularization_weight_map(const stk::Volume& map);
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    /// Set masks.
    void set_fixed_mask(const stk::VolumeFloat& fixed_mask);
    void set_moving_mask(const stk::VolumeFloat& moving_mask);

    /// Sets fixed and moving landmarks.
    void set_landmarks(const std::vector<float3>& fixed_landmarks,
                       const std::vector<float3>& moving_landmarks);

    /// Sets mask and values for constraints
    void set_voxel_constraints(const stk::VolumeUChar& mask, const stk::VolumeFloat3& values);

    /// Runs the registration.
    /// Returns the resulting deformation field or an invalid volume if registration failed.
    stk::Volume execute();

    /// Builds a binary function for the specified pyramid level
    void build_regularizer(int level, Regularizer& binary_fn);

    /// Builds a unary function for the specified pyramid level
    void build_unary_function(int level, UnaryFunction& unary_fn);

    /// Returns the deformation field at the specified pyramid level
    stk::Volume deformation_field(int level);

private:
    /// Determines the neighborhood to apply for the given pyramid level.
    /// The neighborhood is a set of unit vectors in which directions
    /// to apply the optimization. E.g. ((-1,0,0), (1,0,0), (0,-1,0), ...)
    std::vector<int3> determine_neighborhood(int level) const;

#ifdef DF_OUTPUT_DEBUG_VOLUMES
    /// Upsamples the deformation field at the given level and saves it
    void upsample_and_save(int level);

    /// Saves the complete volume pyramids
    void save_volume_pyramid();
#endif

    Settings _settings;

    std::vector<VolumePyramid> _fixed_pyramids;
    std::vector<VolumePyramid> _moving_pyramids;
    VolumePyramid _deformation_pyramid;

    VolumePyramid _fixed_mask_pyramid;
    VolumePyramid _moving_mask_pyramid;

    std::vector<float3> _fixed_landmarks;
    std::vector<float3> _moving_landmarks;

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    VolumePyramid _regularization_weight_map;
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    AffineTransform _affine_transform;

    VolumePyramid _constraints_pyramid;
    VolumePyramid _constraints_mask_pyramid;

};
