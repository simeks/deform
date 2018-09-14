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

void GpuRegistrationEngine::build_unary_function(int level, GpuUnaryFunction& unary_fn)
{
    ASSERT(_fixed_landmarks.size() == 0 && _moving_landmarks.size() == 0);
    ASSERT(!_constraints_mask_pyramid.volume(level).valid());

    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
        stk::GpuVolume fixed;
        stk::GpuVolume moving;

        ASSERT(fixed.voxel_type() == stk::Type_Float);
        ASSERT(moving.voxel_type() == stk::Type_Float);

        if (_fixed_pyramids[i].levels() > 0)
            fixed = _fixed_pyramids[i].volume(level);
        if (_moving_pyramids[i].levels() > 0)
            moving = _moving_pyramids[i].volume(level);

        if (!fixed.valid() || !moving.valid())
            continue; // Skip empty slots

        ASSERT(fixed.voxel_type() == moving.voxel_type());
        for (auto& fn : _settings.image_slots[i].cost_functions) {
            if (Settings::ImageSlot::CostFunction_SSD == fn.function) {
                if (!fn.parameters.empty()) {
                    throw std::invalid_argument("[GPU] SSDFunction: unrecognised parameter "
                                                "'" + fn.parameters.begin()->first + "' with value '" 
                                                + fn.parameters.begin()->second + "'");
                }

                unary_fn.add_function(
                    std::make_unique<GpuCostFunction_SSD>(fixed, moving),
                    fn.weight
                );
            }
            else if (Settings::ImageSlot::CostFunction_NCC == fn.function) {
                int radius = 2;

                for (const auto& [k, v] : fn.parameters) {
                    if (k == "radius") {
                        radius = str_to_num<int>("NCCFunction", k, v);
                    }
                    else {
                        throw std::invalid_argument("[GPU] NCCFunction: unrecognised parameter "
                                                    "'" + k + "' with value '" + v + "'");
                    }
                }

                unary_fn.add_function(
                    std::make_unique<GpuCostFunction_NCC>(fixed, moving, radius),
                    fn.weight
                );
            }
            else {
                FATAL() << "[GPU] Unsupported cost function (slot: " << i << ")";
            }
        }
    }
}
void GpuRegistrationEngine::build_binary_function(int level, GpuBinaryFunction& binary_fn)
{
    binary_fn.set_fixed_spacing(_fixed_pyramids[0].volume(level).spacing());
    binary_fn.set_regularization_weight(_settings.levels[level].regularization_weight);

    // Clone the def, because the current copy will be changed when executing the optimizer
    binary_fn.set_initial_displacement(_deformation_pyramid.volume(level).clone());
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
    FATAL_IF(fixed.voxel_type() != stk::Type_Float || 
             moving.voxel_type() == stk::Type_Float)
        << "Unsupported format";

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
    LOG(Info) << "Running GPU supported registration";

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

            GpuUnaryFunction unary_fn;
            build_unary_function(l, unary_fn);

            GpuBinaryFunction binary_fn;
            build_regularizer(l, binary_fn);

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
                filters::gpu::upsample_vectorfield(def, upsampled_dims)
            );

        }
        else {
            _deformation_pyramid.set_volume(0, def);
        }
    }

    return volume_float4_to_float3(_deformation_pyramid.volume(0).download());
}
