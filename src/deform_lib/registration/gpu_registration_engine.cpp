#include "gpu_registration_engine.h"
#include "gpu/cost_functions/cost_function.h"
#include "hybrid_graph_cut_optimizer.h"

#include "../filters/gpu/resample.h"

#include <stk/cuda/stream.h>
#include <stk/image/gpu_volume.h>

#include <omp.h>

namespace {
    // TODO: Duplicate from registration_engine.cpp
    template<typename T>
    static T str_to_num(const std::string& f, const std::string& k, const std::string& v) {
        try {
            return static_cast<T>(std::stod(v));
        }
        catch (std::invalid_argument&) {
            throw std::invalid_argument(f + ": unrecognised value "
                                        "'" + v + "' for parameter '" + k + "'");
        }
    }

    stk::VolumeFloat4 volume_float3_to_float4(const stk::VolumeFloat3& df)
    {
        dim3 dims = df.size();
        stk::VolumeFloat4 out(dims);
        out.copy_meta_from(df);

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
        out.copy_meta_from(df);

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
    using FunctionPtr = std::unique_ptr<GpuSubFunction>;

    for (int i = 0; i < (int) _fixed_pyramids.size(); ++i) {
        stk::GpuVolume fixed;
        stk::GpuVolume moving;

        if (_fixed_pyramids[i].levels() > 0)
            fixed = _fixed_pyramids[i].volume(level);
        if (_moving_pyramids[i].levels() > 0)
            moving = _moving_pyramids[i].volume(level);

        if (!fixed.valid() || !moving.valid())
            continue; // Skip empty slots

        ASSERT(fixed.voxel_type() == stk::Type_Float);
        ASSERT(moving.voxel_type() == stk::Type_Float);

        ASSERT(fixed.voxel_type() == moving.voxel_type());
        for (auto& fn : _settings.image_slots[i].cost_functions) {
            if (Settings::ImageSlot::CostFunction_SSD == fn.function) {
                if (!fn.parameters.empty()) {
                    throw std::invalid_argument("[GPU] SSDFunction: unrecognised parameter "
                                                "'" + fn.parameters.begin()->first + "' with value '"
                                                + fn.parameters.begin()->second + "'");
                }

                FunctionPtr function = std::make_unique<GpuCostFunction_SSD>(fixed, moving);

                if (_fixed_mask_pyramid.levels() > 0) {
                    function->set_fixed_mask(_fixed_mask_pyramid.volume(level));
                }
                if (_moving_mask_pyramid.levels() > 0) {
                    function->set_moving_mask(_moving_mask_pyramid.volume(level));
                }

                unary_fn.add_function(function, fn.weight);
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

                FunctionPtr function = std::make_unique<GpuCostFunction_NCC>(fixed, moving, radius);

                if (_fixed_mask_pyramid.levels() > 0) {
                    function->set_fixed_mask(_fixed_mask_pyramid.volume(level));
                }
                if (_moving_mask_pyramid.levels() > 0) {
                    function->set_moving_mask(_moving_mask_pyramid.volume(level));
                }

                unary_fn.add_function(function, fn.weight);
            }
            else {
                FATAL() << "[GPU] Unsupported cost function (slot: " << i << ")";
            }
        }
    }

    if (_fixed_landmarks.size() > 0 && level >= _settings.landmarks_stop_level) {
        ASSERT(_fixed_landmarks.size() == _moving_landmarks.size());

        auto& fixed = _fixed_pyramids[0].volume(level);
        FunctionPtr f = std::make_unique<GpuCostFunction_Landmarks>(
                _fixed_landmarks,
                _moving_landmarks,
                fixed,
                _settings.levels[level].landmarks_decay
                );
        unary_fn.add_function(f, _settings.levels[level].landmarks_weight);
    }
}
void GpuRegistrationEngine::build_binary_function(int level, GpuBinaryFunction& binary_fn)
{
    binary_fn.set_fixed_spacing(_fixed_pyramids[0].volume(level).spacing());
    binary_fn.set_regularization_weight(_settings.levels[level].regularization_weight);
    binary_fn.set_regularization_exponent(_settings.levels[level].regularization_exponent);

    // Clone the def, because the current copy will be changed when executing the optimizer
    binary_fn.set_initial_displacement(_deformation_pyramid.volume(level).clone());
}

GpuRegistrationEngine::GpuRegistrationEngine(const Settings& settings) :
    _settings(settings),
    _worker_pool(omp_get_max_threads()-1) // Consider main thread as a worker
{
    // Guess from settings, it will be resized later if too small
    _fixed_pyramids.resize(settings.image_slots.size());
    _moving_pyramids.resize(settings.image_slots.size());

    _deformation_pyramid.set_level_count(_settings.num_pyramid_levels);

    // Create CUDA streams
    _stream_pool.resize(4);
}
GpuRegistrationEngine::~GpuRegistrationEngine()
{
}

void GpuRegistrationEngine::set_initial_deformation(const stk::Volume& def)
{
    // GPU prefers float4 over float3
    ASSERT(def.voxel_type() == stk::Type_Float4 || def.voxel_type() == stk::Type_Float3);
    ASSERT(_settings.num_pyramid_levels);

    // Upload
    stk::GpuVolume gpu_def;
    if (def.voxel_type() == stk::Type_Float3)
        gpu_def = volume_float3_to_float4(def);
    else
        gpu_def = def;

    _deformation_pyramid.build_from_base(gpu_def, filters::gpu::downsample_vectorfield_by_2);
}
void GpuRegistrationEngine::set_image_pair(
        int i,
        const stk::Volume& fixed,
        const stk::Volume& moving)
{
    ASSERT(_fixed_pyramids.size() == _moving_pyramids.size());
    FATAL_IF(fixed.voxel_type() != stk::Type_Float ||
             moving.voxel_type() != stk::Type_Float)
        << "Unsupported format";

    if (i >= (int) _fixed_pyramids.size()) {
        _fixed_pyramids.resize(i + 1);
        _moving_pyramids.resize(i + 1);
    }

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
    ASSERT(fixed_landmarks.size() == moving_landmarks.size());
    _fixed_landmarks = fixed_landmarks;
    _moving_landmarks = moving_landmarks;
}

void GpuRegistrationEngine::set_fixed_mask(const stk::VolumeFloat& fixed_mask)
{
    _fixed_mask_pyramid.set_level_count(_settings.num_pyramid_levels);
    _fixed_mask_pyramid.build_from_base(fixed_mask, filters::gpu::downsample_volume_by_2);
}

void GpuRegistrationEngine::set_moving_mask(const stk::VolumeFloat& moving_mask)
{
    _moving_mask_pyramid.set_level_count(_settings.num_pyramid_levels);
    _moving_mask_pyramid.build_from_base(moving_mask, filters::gpu::downsample_volume_by_2);
}

void GpuRegistrationEngine::set_voxel_constraints(const stk::VolumeUChar& mask,
                                                  const stk::VolumeFloat3& values)
{
    (void) mask; (void) values;

    FATAL() << "Not implemented";
}

stk::Volume GpuRegistrationEngine::execute()
{
    LOG(Info) << "Running GPU supported registration";

    if (!_deformation_pyramid.volume(0).valid()) {
        // No initial deformation, create a field with all zeros

        stk::GpuVolume base = _fixed_pyramids[0].volume(0);

        // TODO: Until we get a proper GpuVolume::fill()
        stk::VolumeFloat4 initial(base.size(), float4{0, 0, 0, 0});
        initial.set_origin(base.origin());
        initial.set_spacing(base.spacing());
        initial.set_direction(base.direction());

        set_initial_deformation(initial);
    }

    for (int l = _settings.num_pyramid_levels-1; l >= 0; --l) {
        stk::GpuVolume df = _deformation_pyramid.volume(l);

        if (l >= _settings.pyramid_stop_level) {
            LOG(Info) << "Performing registration level " << l;

            GpuUnaryFunction unary_fn;
            build_unary_function(l, unary_fn);

            GpuBinaryFunction binary_fn;
            build_binary_function(l, binary_fn);

            HybridGraphCutOptimizer optimizer(
                _settings.levels[l],
                unary_fn,
                binary_fn,
                df,
                _worker_pool,
                _stream_pool
            );

            optimizer.execute();
        }
        else {
            LOG(Info) << "Skipping level " << l;
        }

        if (l != 0) {
            dim3 upsampled_dims = _deformation_pyramid.volume(l - 1).size();
            _deformation_pyramid.set_volume(l - 1,
                filters::gpu::upsample_vectorfield(df, upsampled_dims)
            );

        }
        else {
            _deformation_pyramid.set_volume(0, df);
        }
    }

    return volume_float4_to_float3(_deformation_pyramid.volume(0).download());
}
