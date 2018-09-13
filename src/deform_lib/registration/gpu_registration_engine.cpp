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

void GpuRegistrationEngine::set_initial_deformation(const stk::GpuVolume& def)
{
    // GPU prefers float4 over float3
    ASSERT(def.voxel_type() == stk::Type_Float4);
    ASSERT(_settings.num_pyramid_levels);
    
  //  _deformation_pyramid.build_from_base(def, filters::gpu::downsample_vectorfield_by_2);
}
void GpuRegistrationEngine::set_image_pair(
        int i, 
        const stk::GpuVolume& fixed, 
        const stk::GpuVolume& moving,
        stk::GpuVolume (*downsample_fn)(const stk::GpuVolume&))
{
    ASSERT(i < DF_MAX_IMAGE_PAIR_COUNT);
    
    _fixed_pyramids[i].set_level_count(_settings.num_pyramid_levels);
    _moving_pyramids[i].set_level_count(_settings.num_pyramid_levels);
    
    _fixed_pyramids[i].build_from_base(fixed, downsample_fn);
    _moving_pyramids[i].build_from_base(moving, downsample_fn);
}
stk::GpuVolume GpuRegistrationEngine::execute()
{
    return stk::GpuVolume();
}
