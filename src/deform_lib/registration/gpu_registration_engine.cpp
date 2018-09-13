#include "gpu_registration_engine.h"

#include "../filters/gpu/resample.h"

#include <stk/image/gpu_volume.h>

GpuRegistrationEngine::GpuRegistrationEngine(const Settings& settings) :
    _settings(settings)
{
    _fixed_pyramids.resize(DF_MAX_IMAGE_PAIR_COUNT);
    _moving_pyramids.resize(DF_MAX_IMAGE_PAIR_COUNT);

    _deformation_pyramid.set_level_count(_settings.num_pyramid_levels);
}
GpuRegistrationEngine::~GpuRegistrationEngine()
{
}

void GpuRegistrationEngine::set_initial_deformation(const stk::Volume& def)
{
    // GPU prefers float4 over float3
    ASSERT(def.voxel_type() == stk::Type_Float4);
    ASSERT(_settings.num_pyramid_levels);
    
  //  _deformation_pyramid.build_from_base(def, filters::gpu::downsample_vectorfield_by_2);
}
void GpuRegistrationEngine::set_image_pair(
        int i, 
        const stk::Volume& fixed, 
        const stk::Volume& moving)
{
    ASSERT(i < DF_MAX_IMAGE_PAIR_COUNT);
    
    _fixed_pyramids[i].set_level_count(_settings.num_pyramid_levels);
    _moving_pyramids[i].set_level_count(_settings.num_pyramid_levels);
    
    auto downsample_fn = filters::gpu::downsample_volume_by_2;

    // Upload
    stk::GpuVolume gpu_fixed(fixed);
    stk::GpuVolume gpu_moving(moving);

    _fixed_pyramids[i].build_from_base(gpu_fixed, downsample_fn);
    _moving_pyramids[i].build_from_base(gpu_moving, downsample_fn);
}

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
void GpuRegistrationEngine::set_regularization_weight_map(const stk::Volume& map)
{
    FATAL() << "Not implemented";
}
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

void GpuRegistrationEngine::set_landmarks(const std::vector<float3>& fixed_landmarks,
                                          const std::vector<float3>& moving_landmarks)
{
    FATAL() << "Not implemented";
}

void GpuRegistrationEngine::set_voxel_constraints(const stk::VolumeUChar& mask, 
                                                  const stk::VolumeFloat3& values)
{
    FATAL() << "Not implemented";
}

stk::Volume GpuRegistrationEngine::execute()
{
    return stk::Volume();
}
