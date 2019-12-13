#pragma once

#include "../config.h"
#include "affine_transform.h"
#include "settings.h"
#include "gpu_volume_pyramid.h"
#include "worker_pool.h"

#include <stk/image/volume.h>

namespace stk { namespace cuda {
    class Stream;
}}

class GpuBinaryFunction;
class GpuUnaryFunction;
class GpuRegistrationEngine
{
public:
    GpuRegistrationEngine(const Settings& settings);
    ~GpuRegistrationEngine();

    void set_initial_displacement_field(const stk::Volume& def);
    void set_affine_transform(const AffineTransform& affine_transform);

    void set_image_pair(
        int i,
        const stk::Volume& fixed,
        const stk::Volume& moving
    );

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

private:
    /// Determines the neighborhood to apply for the given pyramid level.
    /// The neighborhood is a set of unit vectors in which directions
    /// to apply the optimization. E.g. ((-1,0,0), (1,0,0), (0,-1,0), ...)
    std::vector<int3> determine_neighborhood(int level) const;
    
    /// Builds a unary function for the specified pyramid level
    void build_unary_function(int level, GpuUnaryFunction& unary_fn);

    /// Builds a binary function for the specified pyramid level
    void build_binary_function(int level, GpuBinaryFunction& binary_fn);

    Settings _settings;

    std::vector<GpuVolumePyramid> _fixed_pyramids;
    std::vector<GpuVolumePyramid> _moving_pyramids;
    GpuVolumePyramid _deformation_pyramid;

    GpuVolumePyramid _fixed_mask_pyramid;
    GpuVolumePyramid _moving_mask_pyramid;

    std::vector<float3> _fixed_landmarks;
    std::vector<float3> _moving_landmarks;

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    GpuVolumePyramid _regularization_weight_map;
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    AffineTransform _affine_transform;

    WorkerPool _worker_pool;
    std::vector<stk::cuda::Stream> _stream_pool;
};
