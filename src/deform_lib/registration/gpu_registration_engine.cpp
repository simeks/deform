#include "deform_lib/make_unique.h"
#include "deform_lib/profiler/profiler.h"

#ifdef DF_ENABLE_GCO
    #include "deform_lib/solver/gco_solver.h"
#endif
#ifdef DF_ENABLE_GRIDCUT
    #include "deform_lib/solver/gridcut_solver.h"
#endif
#include "deform_lib/solver/icm_solver.h"

#include "gpu_registration_engine.h"
#include "gpu/cost_functions/binary_function.h"
#include "gpu/cost_functions/cross_correlation.h"
#include "gpu/cost_functions/landmarks.h"
#include "gpu/cost_functions/squared_distance.h"
#include "gpu/cost_functions/unary_function.h"
#include "gpu/gpu_displacement_field.h"
#include "hybrid_graph_cut_optimizer.h"

#include "../filters/gpu/resample.h"

#include <stk/cuda/stream.h>
#include <stk/image/gpu_volume.h>

#include <omp.h>

namespace {
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
    using FunctionPtr = std::unique_ptr<GpuCostFunction>;

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

                FunctionPtr function = make_unique<GpuCostFunction_SSD>(fixed, moving);

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

                for (const auto& p : fn.parameters) {
                    if (p.first == "radius") {
                        radius = str_to_num<int>("NCCFunction", p.first, p.second);
                    }
                    else {
                        throw std::invalid_argument("[GPU] NCCFunction: unrecognised parameter "
                                                    "'" + p.first + "' with value '" + p.second + "'");
                    }
                }

                FunctionPtr function = make_unique<GpuCostFunction_NCC>(fixed, moving, radius);

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
        FunctionPtr f = make_unique<GpuCostFunction_Landmarks>(
                fixed,
                _fixed_landmarks,
                _moving_landmarks,
                _settings.levels[level].landmarks_decay
                );
        unary_fn.add_function(f, _settings.levels[level].landmarks_weight);
    }
}
void GpuRegistrationEngine::build_binary_function(int level, GpuBinaryFunction& binary_fn)
{
    binary_fn.set_fixed_spacing(_fixed_pyramids[0].volume(level).spacing());
    binary_fn.set_regularization_weight(_settings.levels[level].regularization_weight);
    binary_fn.set_regularization_scale(_settings.levels[level].regularization_scale);
    binary_fn.set_regularization_exponent(_settings.levels[level].regularization_exponent);
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
    _stream_pool.resize(5);
}
GpuRegistrationEngine::~GpuRegistrationEngine()
{
}

void GpuRegistrationEngine::set_initial_displacement_field(const stk::Volume& def)
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
void GpuRegistrationEngine::set_affine_transform(const AffineTransform& affine_transform)
{
    _affine_transform = affine_transform;
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

    stk::GpuVolume gpu_fixed(fixed.size(), fixed.voxel_type());
    stk::GpuVolume gpu_moving(moving.size(), moving.voxel_type());

    {
        PROFILER_SCOPE("upload_image_pair", 0xFF532439);
        gpu_fixed.upload(fixed);
        gpu_moving.upload(moving);
    }

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

        stk::VolumeFloat4 initial(base.size(), float4{0, 0, 0, 0});
        initial.set_origin(base.origin());
        initial.set_spacing(base.spacing());
        initial.set_direction(base.direction());
        
        set_initial_displacement_field(initial);
    }

    for (int l = _settings.num_pyramid_levels-1; l >= 0; --l) {
        GpuDisplacementField df(_deformation_pyramid.volume(l), _affine_transform);

        std::vector<int3> neighborhood = determine_neighborhood(l);

        if (l >= _settings.pyramid_stop_level) {
            LOG(Info) << "Performing registration level " << l;

            GpuUnaryFunction unary_fn;
            build_unary_function(l, unary_fn);

            GpuBinaryFunction binary_fn;
            build_binary_function(l, binary_fn);

            if (_settings.solver == Settings::Solver_ICM) {
                HybridGraphCutOptimizer<ICMSolver<double>> optimizer(
                    neighborhood,
                    _settings.levels[l],
                    _settings.update_rule,
                    unary_fn,
                    binary_fn,
                    df,
                    _worker_pool,
                    _stream_pool
                );
                optimizer.execute();
            }
#if defined(DF_ENABLE_GCO)
            else if (_settings.solver == Settings::Solver_GCO) {
                HybridGraphCutOptimizer<GCOSolver<double>> optimizer(
                    neighborhood,
                    _settings.levels[l],
                    _settings.update_rule,
                    unary_fn,
                    binary_fn,
                    df,
                    _worker_pool,
                    _stream_pool
                );
                optimizer.execute();
            }
#endif
#if defined(DF_ENABLE_GRIDCUT)
            else if (_settings.solver == Settings::Solver_GridCut) {
                HybridGraphCutOptimizer<GridCutSolver<double>> optimizer(
                    neighborhood,
                    _settings.levels[l],
                    _settings.update_rule
                    unary_fn,
                    binary_fn,
                    df,
                    _worker_pool,
                    _stream_pool
                );
                optimizer.execute();
            }
#endif

        }
        else {
            LOG(Info) << "Skipping level " << l;
        }

        stk::GpuVolume vf = df.volume();
        if (l != 0) {
            dim3 upsampled_dims = _deformation_pyramid.volume(l - 1).size();
            _deformation_pyramid.set_volume(l - 1,
                filters::gpu::upsample_vectorfield(vf, upsampled_dims)
            );
        }
        else {
            _deformation_pyramid.set_volume(0, vf);
        }
    }

    return volume_float4_to_float3(
        cuda::compute_displacement_field(
            _deformation_pyramid.volume(0),
            _affine_transform
        ).download()
    );
}
std::vector<int3> GpuRegistrationEngine::determine_neighborhood(int level) const
{
    // Identical to RegistrationEngine::determine_neighborhood with the exception
    //  of working on GpuVolumePyramid

    std::vector<int3> neighborhood;

    dim3 dim_size {0, 0, 0};

    for (int i = 0; i < (int) _fixed_pyramids.size(); ++i) {
        stk::GpuVolume fixed;
        stk::GpuVolume moving;

        if (_fixed_pyramids[i].levels() > 0)
            fixed = _fixed_pyramids[i].volume(level);
        if (_moving_pyramids[i].levels() > 0)
            moving = _moving_pyramids[i].volume(level);

        dim_size = {
            std::max(dim_size.x, fixed.size().x),
            std::max(dim_size.y, fixed.size().y),
            std::max(dim_size.z, fixed.size().z)
        };
        
        dim_size = {
            std::max(dim_size.x, moving.size().x),
            std::max(dim_size.y, moving.size().y),
            std::max(dim_size.z, moving.size().z)
        };
    }

    if (dim_size.x > 1) {
        neighborhood.push_back({1, 0, 0});
        neighborhood.push_back({-1, 0, 0});
    }

    if (dim_size.y > 1) {
        neighborhood.push_back({0, 1, 0});
        neighborhood.push_back({0, -1, 0});
    }

    if (dim_size.z > 1) {
        neighborhood.push_back({0, 0, 1});
        neighborhood.push_back({0, 0, -1});
    }

    return neighborhood;
}
