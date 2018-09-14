#include "gpu_registration_engine.h"
#include "gpu/cost_function.h"

#include "../filters/gpu/resample.h"

#include <stk/image/gpu_volume.h>

namespace {
    stk::VolumeFloat4 volume_float3_to_float4(const stk::VolumeFloat3& df)
    {
        dim3 dims = df.size();
        stk::VolumeFloat4 out(dims);

        for (int z = 0; z < (int)dims.z; ++z) {
        for (int y = 0; y < (int)dims.y; ++y) {
        for (int x = 0; x < (int)dims.x; ++x) {
            out(x,y,z) = float4{df(x,y,z).x, df(x,y,z).y, df(x,y,z).z, 0.0f};
        }}}
        
        return out;
    }
    stk::VolumeFloat3 volume_float4_to_float3(const stk::VolumeFloat4& df)
    {
        dim3 dims = df.size();
        stk::VolumeFloat3 out(dims);

        for (int z = 0; z < (int)dims.z; ++z) {
        for (int y = 0; y < (int)dims.y; ++y) {
        for (int x = 0; x < (int)dims.x; ++x) {
            out(x,y,z) = float3{df(x,y,z).x, df(x,y,z).y, df(x,y,z).z};
        }}}
        
        return out;
    }
}

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
    
    // Upload
    stk::GpuVolume gpu_def(volume_float3_to_float4(def));

    _deformation_pyramid.build_from_base(gpu_def, filters::gpu::downsample_vectorfield_by_2);
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
    map;

    FATAL() << "Not implemented";
}
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

void GpuRegistrationEngine::set_landmarks(const std::vector<float3>& fixed_landmarks,
                                          const std::vector<float3>& moving_landmarks)
{
    fixed_landmarks; moving_landmarks;
    
    FATAL() << "Not implemented";
}

void GpuRegistrationEngine::set_voxel_constraints(const stk::VolumeUChar& mask, 
                                                  const stk::VolumeFloat3& values)
{
    mask; values;

    FATAL() << "Not implemented";
}

stk::Volume GpuRegistrationEngine::execute()
{
    if (!_deformation_pyramid.volume(0).valid()) {
        // No initial deformation, create a field with all zeros
        
        stk::GpuVolume base = _fixed_pyramids[0].volume(0);

        // TODO: Until we get a proper GpuVolume::fill()
        stk::VolumeFloat4 initial(base.size(), float4{0});
        initial.set_origin(base.origin());
        initial.set_spacing(base.spacing());
        
        stk::GpuVolume gpu_initial(initial);
        set_initial_deformation(gpu_initial);
    }

    for (int l = _settings.num_pyramid_levels-1; l >= 0; --l) {
        stk::GpuVolume df = _deformation_pyramid.volume(l);

        if (l >= _settings.pyramid_stop_level) {
            LOG(Info) << "Performing registration level " << l;

            UnaryFunction unary_fn;
            build_unary_function(l, unary_fn);

            Regularizer binary_fn;
            build_regularizer(l, binary_fn);

            if (_constraints_mask_pyramid.volume(l).valid())
            {
                // Fix constrained voxels by updating the initial deformation field
                constrain_deformation_field(
                    def,
                    _constraints_mask_pyramid.volume(l),
                    _constraints_pyramid.volume(l)
                );
            }
            
            BlockedGraphCutOptimizer<UnaryFunction, Regularizer> optimizer(
                _settings.levels[l].block_size,
                _settings.levels[l].block_energy_epsilon,
                _settings.levels[l].max_iteration_count
            );

            optimizer.execute(unary_fn, binary_fn, _settings.levels[l].step_size, def);
        }
        else {
            LOG(Info) << "Skipping level " << l;
        }
        
        if (l != 0) {
            dim3 upsampled_dims = _deformation_pyramid.volume(l - 1).size();
            _deformation_pyramid.set_volume(l - 1,
                filters::upsample_vectorfield(def, upsampled_dims)
            );

        }
        else {
            _deformation_pyramid.set_volume(0, def);
        }
    }


    stk::Volume out = volume_float4_to_float3(_deformation_pyramid.volume(0).download());
    return out;
}
