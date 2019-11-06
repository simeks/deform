#include "../config.h"
#include "../filters/resample.h"
#include "../make_unique.h"

#ifdef DF_ENABLE_GCO
    #include "../solver/gco_solver.h"
#endif
#ifdef DF_ENABLE_GRIDCUT
    #include "../solver/gridcut_solver.h"
#endif
#include "../solver/icm_solver.h"

#include "cost_functions/cost_function.h"

#include "blocked_graph_cut_optimizer.h"
#include "displacement_field.h"
#include "registration_engine.h"
#include "transform.h"

#include <stk/common/assert.h>
#include <stk/common/log.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>

#include <sstream>
#include <string>

#include "voxel_constraints.h"


namespace
{
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

    void constrain_deformation_field(DisplacementField& df,
        const stk::VolumeUChar& mask, const stk::VolumeFloat3& values)
    {
        ASSERT(df.size() == mask.size() && df.size() == values.size());
        dim3 dims = df.size();
        for (int z = 0; z < int(dims.z); ++z) {
            for (int y = 0; y < int(dims.y); ++y) {
                for (int x = 0; x < int(dims.x); ++x) {
                    if (mask(x, y, z) > 0) {
                        df.set(int3{x,y,z}, values(x, y, z));
                    }
                }
            }
        }
    }

    template<typename T>
    std::unique_ptr<SubFunction> ssd_function_factory(
        const stk::Volume& fixed,
        const stk::Volume& moving,
        const stk::VolumeFloat& moving_mask,
        const std::map<std::string, std::string>& parameters
    )
    {
        if (!parameters.empty()) {
            throw std::invalid_argument("SSDFunction: unrecognised parameter "
                                        "'" + parameters.begin()->first + "' with value '"
                                        + parameters.begin()->second + "'");
        }

        std::unique_ptr<SubFunction> function = make_unique<SquaredDistanceFunction<T>>(fixed, moving);
        if (moving_mask.valid()) {
            function->set_moving_mask(moving_mask);
        }

        return function;
    }

    template<typename T>
    std::unique_ptr<SubFunction> ncc_function_factory(
        const stk::Volume& fixed,
        const stk::Volume& moving,
        const stk::VolumeFloat& moving_mask,
        const std::map<std::string, std::string>& parameters
    )
    {
        int radius = 2;
        std::string window = "sphere";

        for (const auto& p : parameters) {
            if (p.first == "radius") {
                radius = str_to_num<int>("NCCFunction", p.first, p.second);
            }
            else if (p.first == "window") {
                if (p.second != "cube" && p.second != "sphere") {
                    throw std::invalid_argument("NCCFunction: invalid value '" + p.second +
                                                "' for parameter '" + p.first + "'");
                }
                window = p.second;
            }
            else {
                throw std::invalid_argument("NCCFunction: unrecognised parameter "
                                            "'" + p.first + "' with value '" + p.second + "'");
            }
        }

        std::unique_ptr<SubFunction> function;
        if ("sphere" == window) {
            function = make_unique<NCCFunction_sphere<T>>(fixed, moving, radius);
            if (moving_mask.valid()) {
                function->set_moving_mask(moving_mask);
            }
        }
        else if ("cube" == window) {
            function = make_unique<NCCFunction_cube<T>>(fixed, moving, radius);
            if (moving_mask.valid()) {
                function->set_moving_mask(moving_mask);
            }
        }
        else {
            throw std::runtime_error("NCCFunction: there is a bug in the selection of the window.");
        }

        return function;
    }

    template<typename T>
    std::unique_ptr<SubFunction> mi_function_factory(
        const stk::Volume& fixed,
        const stk::Volume& moving,
        const stk::VolumeFloat& moving_mask,
        const std::map<std::string, std::string>& parameters
    )
    {
        int bins = 256;
        double sigma = 4.5;
        int update_interval = 1;
        transform::Interp interpolator = transform::Interp_NN;

        for (const auto& p : parameters) {
            if (p.first == "bins") {
                bins = str_to_num<int>("MIFunction", p.first, p.second);
            }
            else if (p.first == "sigma") {
                sigma = str_to_num<double>("MIFunction", p.first, p.second);
            }
            else if (p.first == "update_interval") {
                update_interval = str_to_num<int>("MIFunction", p.first, p.second);
            }
            else if (p.first == "interpolator") {
                if (p.second == "linear") {
                    interpolator = transform::Interp_Linear;
                }
                else if (p.second == "nearest" || p.second == "nn") {
                    interpolator = transform::Interp_NN;
                }
                else {
                    throw std::invalid_argument("MIFunction: invalid interpolator '" + p.second + "'");
                }
            }
            else {
                throw std::invalid_argument("MIFunction: unrecognised parameter "
                                            "'" + p.first + "' with p.secondalue '" + p.second + "'");
            }
        }

        std::unique_ptr<SubFunction> function =
            make_unique<MIFunction<T>>(fixed, moving, bins, sigma, update_interval, interpolator);
        if (moving_mask.valid()) {
            function->set_moving_mask(moving_mask);
        }

        return function;
    }

    template<typename T>
    std::unique_ptr<SubFunction> gradient_ssd_function_factory(
        const stk::Volume& fixed,
        const stk::Volume& moving,
        const stk::VolumeFloat& moving_mask,
        const std::map<std::string, std::string>& parameters
    )
    {
        float sigma = 0.0f;

        for (const auto& p : parameters) {
            if (p.first == "sigma") {
                sigma = str_to_num<float>("GradientSSDFunction", p.first, p.second);
            }
            else {
                throw std::invalid_argument("GradientSSDFunction: unrecognised parameter "
                                            "'" + p.first + "' with value '" + p.second + "'");
            }
        }

        std::unique_ptr<SubFunction> function =
            make_unique<GradientSSDFunction<T>>(fixed, moving, sigma);
        if (moving_mask.valid()) {
            function->set_moving_mask(moving_mask);
        }

        return function;
    }
}

void RegistrationEngine::build_regularizer(int level, Regularizer& binary_fn)
{
    binary_fn.set_fixed_spacing(_fixed_pyramids[0].volume(level).spacing());
    binary_fn.set_regularization_weight(_settings.levels[level].regularization_weight);
    binary_fn.set_regularization_scale(_settings.levels[level].regularization_scale);
    binary_fn.set_regularization_exponent(_settings.levels[level].regularization_exponent);

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    if (_regularization_weight_map.volume(level).valid())
        binary_fn.set_weight_map(_regularization_weight_map.volume(level));
#endif
}

void RegistrationEngine::build_unary_function(int level, UnaryFunction& unary_fn)
{
    typedef std::unique_ptr<SubFunction> (*FactoryFn)(
        const stk::Volume&,
        const stk::Volume&,
        const stk::VolumeFloat&,
        const std::map<std::string, std::string>&
    );

    // nullptr => not supported
    FactoryFn ssd_factory[] = {
        nullptr, // Type_Unknown

        ssd_function_factory<char>, // Type_Char
        nullptr, // Type_Char2
        nullptr, // Type_Char3
        nullptr, // Type_Char4

        ssd_function_factory<uint8_t>, // Type_UChar
        nullptr, // Type_UChar2
        nullptr, // Type_UChar3
        nullptr, // Type_UChar4

        ssd_function_factory<short>, // Type_Short
        nullptr, // Type_Short2
        nullptr, // Type_Short3
        nullptr, // Type_Short4

        ssd_function_factory<uint16_t>, // Type_UShort
        nullptr, // Type_UShort2
        nullptr, // Type_UShort3
        nullptr, // Type_UShort4

        ssd_function_factory<int>, // Type_Int
        nullptr, // Type_Int2
        nullptr, // Type_Int3
        nullptr, // Type_Int4

        ssd_function_factory<uint32_t>, // Type_UInt
        nullptr, // Type_UInt2
        nullptr, // Type_UInt3
        nullptr, // Type_UInt4

        ssd_function_factory<float>, // Type_Float
        nullptr, // Type_Float2
        nullptr, // Type_Float3
        nullptr, // Type_Float4

        ssd_function_factory<double>, // Type_Double
        nullptr, // Type_Double2
        nullptr, // Type_Double3
        nullptr // Type_Double4
    };
    FactoryFn ncc_factory[] = {
        nullptr, // Type_Unknown

        ncc_function_factory<char>, // Type_Char
        nullptr, // Type_Char2
        nullptr, // Type_Char3
        nullptr, // Type_Char4

        ncc_function_factory<uint8_t>, // Type_UChar
        nullptr, // Type_UChar2
        nullptr, // Type_UChar3
        nullptr, // Type_UChar4

        ncc_function_factory<short>, // Type_Short
        nullptr, // Type_Short2
        nullptr, // Type_Short3
        nullptr, // Type_Short4

        ncc_function_factory<uint16_t>, // Type_UShort
        nullptr, // Type_UShort2
        nullptr, // Type_UShort3
        nullptr, // Type_UShort4

        ncc_function_factory<int>, // Type_Int
        nullptr, // Type_Int2
        nullptr, // Type_Int3
        nullptr, // Type_Int4

        ncc_function_factory<uint32_t>, // Type_UInt
        nullptr, // Type_UInt2
        nullptr, // Type_UInt3
        nullptr, // Type_UInt4

        ncc_function_factory<float>, // Type_Float
        nullptr, // Type_Float2
        nullptr, // Type_Float3
        nullptr, // Type_Float4

        ncc_function_factory<double>, // Type_Double
        nullptr, // Type_Double2
        nullptr, // Type_Double3
        nullptr // Type_Double4
    };
    FactoryFn mi_factory[] = {
        nullptr, // Type_Unknown

        mi_function_factory<char>, // Type_Char
        nullptr, // Type_Char2
        nullptr, // Type_Char3
        nullptr, // Type_Char4

        mi_function_factory<uint8_t>, // Type_UChar
        nullptr, // Type_UChar2
        nullptr, // Type_UChar3
        nullptr, // Type_UChar4

        mi_function_factory<short>, // Type_Short
        nullptr, // Type_Short2
        nullptr, // Type_Short3
        nullptr, // Type_Short4

        mi_function_factory<uint16_t>, // Type_UShort
        nullptr, // Type_UShort2
        nullptr, // Type_UShort3
        nullptr, // Type_UShort4

        mi_function_factory<int>, // Type_Int
        nullptr, // Type_Int2
        nullptr, // Type_Int3
        nullptr, // Type_Int4

        mi_function_factory<uint32_t>, // Type_UInt
        nullptr, // Type_UInt2
        nullptr, // Type_UInt3
        nullptr, // Type_UInt4

        mi_function_factory<float>, // Type_Float
        nullptr, // Type_Float2
        nullptr, // Type_Float3
        nullptr, // Type_Float4

        mi_function_factory<double>, // Type_Double
        nullptr, // Type_Double2
        nullptr, // Type_Double3
        nullptr // Type_Double4
    };
    FactoryFn gradient_ssd_factory[] = {
        nullptr, // Type_Unknown

        gradient_ssd_function_factory<char>, // Type_Char
        nullptr, // Type_Char2
        nullptr, // Type_Char3
        nullptr, // Type_Char4

        gradient_ssd_function_factory<uint8_t>, // Type_UChar
        nullptr, // Type_UChar2
        nullptr, // Type_UChar3
        nullptr, // Type_UChar4

        gradient_ssd_function_factory<short>, // Type_Short
        nullptr, // Type_Short2
        nullptr, // Type_Short3
        nullptr, // Type_Short4

        gradient_ssd_function_factory<uint16_t>, // Type_UShort
        nullptr, // Type_UShort2
        nullptr, // Type_UShort3
        nullptr, // Type_UShort4

        gradient_ssd_function_factory<int>, // Type_Int
        nullptr, // Type_Int2
        nullptr, // Type_Int3
        nullptr, // Type_Int4

        gradient_ssd_function_factory<uint32_t>, // Type_UInt
        nullptr, // Type_UInt2
        nullptr, // Type_UInt3
        nullptr, // Type_UInt4

        gradient_ssd_function_factory<float>, // Type_Float
        nullptr, // Type_Float2
        nullptr, // Type_Float3
        nullptr, // Type_Float4

        gradient_ssd_function_factory<double>, // Type_Double
        nullptr, // Type_Double2
        nullptr, // Type_Double3
        nullptr // Type_Double4
    };

    if (_fixed_mask_pyramid.levels() > 0) {
        unary_fn.set_fixed_mask(_fixed_mask_pyramid.volume(level));
    }

    auto const& moving_mask = _moving_mask_pyramid.levels() > 0 ? _moving_mask_pyramid.volume(level)
                                                                : stk::VolumeFloat();

    for (int i = 0; i < (int) _fixed_pyramids.size(); ++i) {
        stk::Volume fixed;
        stk::Volume moving;

        if (_fixed_pyramids[i].levels() > 0)
            fixed = _fixed_pyramids[i].volume(level);
        if (_moving_pyramids[i].levels() > 0)
            moving = _moving_pyramids[i].volume(level);

        if (!fixed.valid() || !moving.valid())
            continue; // Skip empty slots

        ASSERT(fixed.voxel_type() == moving.voxel_type());
        for (auto& fn : _settings.image_slots[i].cost_functions) {
            if (Settings::ImageSlot::CostFunction_SSD == fn.function) {
                FactoryFn factory = ssd_factory[fixed.voxel_type()];
                if (factory) {
                    unary_fn.add_function(factory(fixed, moving, moving_mask, fn.parameters), fn.weight);
                }
                else {
                    FATAL() << "Unsupported voxel type (" << fixed.voxel_type() << ") "
                            << "for metric 'ssd' "
                            << "(slot: " << i << ")";
                }
            }
            else if (Settings::ImageSlot::CostFunction_NCC == fn.function)
            {
                FactoryFn factory = ncc_factory[fixed.voxel_type()];
                if (factory) {
                    unary_fn.add_function(factory(fixed, moving, moving_mask, fn.parameters), fn.weight);
                }
                else {
                    FATAL() << "Unsupported voxel type (" << fixed.voxel_type() << ") "
                            << "for metric 'ncc' "
                            << "(slot: " << i << ")";
                }
            }
            else if (Settings::ImageSlot::CostFunction_MI == fn.function)
            {
                FactoryFn factory = mi_factory[fixed.voxel_type()];
                if (factory) {
                    unary_fn.add_function(factory(fixed, moving, moving_mask, fn.parameters), fn.weight);
                }
                else {
                    FATAL() << "Unsupported voxel type (" << fixed.voxel_type() << ") "
                            << "for metric 'mi' "
                            << "(slot: " << i << ")";
                }
            }
            else if (Settings::ImageSlot::CostFunction_Gradient_SSD == fn.function)
            {
                FactoryFn factory = gradient_ssd_factory[fixed.voxel_type()];
                if (factory) {
                    unary_fn.add_function(factory(fixed, moving, moving_mask, fn.parameters), fn.weight);
                }
                else {
                    FATAL() << "Unsupported voxel type (" << fixed.voxel_type() << ") "
                            << "for metric 'gradient_ssd' "
                            << "(slot: " << i << ")";
                }
            }
        }
    }

    if (_fixed_landmarks.size() > 0 && level >= _settings.landmarks_stop_level) {
        ASSERT(_fixed_landmarks.size() == _moving_landmarks.size());

        auto& fixed = _fixed_pyramids[0].volume(level);

        unary_fn.add_function(
            make_unique<LandmarksFunction>(
                _fixed_landmarks,
                _moving_landmarks,
                fixed,
                _settings.levels[level].landmarks_decay
            ),
            _settings.levels[level].landmarks_weight
        );
    }

    if (_constraints_mask_pyramid.volume(level).valid()) {
        unary_fn.add_function(
            make_unique<SoftConstraintsFunction>(
                _constraints_mask_pyramid.volume(level),
                _constraints_pyramid.volume(level)
            ),
            _settings.levels[level].constraints_weight
        );
    }
}

RegistrationEngine::RegistrationEngine(const Settings& settings) :
    _settings(settings)
{
    // Guess from settings, it will be resized later if too small
    _fixed_pyramids.resize(settings.image_slots.size());
    _moving_pyramids.resize(settings.image_slots.size());

    _deformation_pyramid.set_level_count(_settings.num_pyramid_levels);

    #ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        _regularization_weight_map.set_level_count(_settings.num_pyramid_levels);
    #endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    _constraints_pyramid.set_level_count(_settings.num_pyramid_levels);
    _constraints_mask_pyramid.set_level_count(_settings.num_pyramid_levels);
}
RegistrationEngine::~RegistrationEngine()
{
}
void RegistrationEngine::set_fixed_mask(const stk::VolumeFloat& fixed_mask)
{
    _fixed_mask_pyramid.set_level_count(_settings.num_pyramid_levels);
    _fixed_mask_pyramid.build_from_base(fixed_mask, filters::downsample_volume_by_2);
}
void RegistrationEngine::set_moving_mask(const stk::VolumeFloat& moving_mask)
{
    _moving_mask_pyramid.set_level_count(_settings.num_pyramid_levels);
    _moving_mask_pyramid.build_from_base(moving_mask, filters::downsample_volume_by_2);
}
void RegistrationEngine::set_initial_deformation(const stk::Volume& def)
{
    ASSERT(def.voxel_type() == stk::Type_Float3); // Only single-precision supported for now
    ASSERT(_settings.num_pyramid_levels);

#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    _deformation_pyramid.build_from_base_with_residual(def, filters::downsample_vectorfield_by_2);
#else
    _deformation_pyramid.build_from_base(def, filters::downsample_vectorfield_by_2);
#endif
}
void RegistrationEngine::set_image_pair(
    int i,
    const stk::Volume& fixed,
    const stk::Volume& moving)
{
    ASSERT(_fixed_pyramids.size() == _moving_pyramids.size());
    if (i >= (int) _fixed_pyramids.size()) {
        _fixed_pyramids.resize(i + 1);
        _moving_pyramids.resize(i + 1);
    }

    _fixed_pyramids[i].set_level_count(_settings.num_pyramid_levels);
    _moving_pyramids[i].set_level_count(_settings.num_pyramid_levels);

    // It's the only available fn for now
    auto downsample_fn = filters::downsample_volume_by_2;

    _fixed_pyramids[i].build_from_base(fixed, downsample_fn);
    _moving_pyramids[i].build_from_base(moving, downsample_fn);
}
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
void RegistrationEngine::set_regularization_weight_map(const stk::Volume& map)
{
    _regularization_weight_map.build_from_base(map, filters::downsample_volume_by_2);
}
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

void RegistrationEngine::set_landmarks(
        const std::vector<float3>& fixed_landmarks,
        const std::vector<float3>& moving_landmarks)
{
    ASSERT(fixed_landmarks.size() == moving_landmarks.size());
    _fixed_landmarks = fixed_landmarks;
    _moving_landmarks = moving_landmarks;
}

void RegistrationEngine::set_voxel_constraints(const stk::VolumeUChar& mask, const stk::VolumeFloat3& values)
{
    voxel_constraints::build_pyramids(
        mask,
        values,
        _settings.num_pyramid_levels,
        _constraints_mask_pyramid,
        _constraints_pyramid
    );
}

stk::Volume RegistrationEngine::execute()
{
    if (!_deformation_pyramid.volume(0).valid()) {
        // No initial deformation, create a field with all zeros

        const stk::Volume& base = _fixed_pyramids[0].volume(0);

        stk::VolumeFloat3 initial(base.size(), float3{0, 0, 0});
        initial.copy_meta_from(base);
        set_initial_deformation(initial);
    }

    #ifdef DF_OUTPUT_DEBUG_VOLUMES
        save_volume_pyramid();
    #endif

    for (int l = _settings.num_pyramid_levels-1; l >= 0; --l) {
        DisplacementField df(_deformation_pyramid.volume(l));

        std::vector<int3> neighborhood = determine_neighborhood(l);

        if (l >= _settings.pyramid_stop_level) {
            LOG(Info) << "Performing registration level " << l;

            Regularizer binary_fn;
            build_regularizer(l, binary_fn);

            if (_constraints_mask_pyramid.volume(l).valid())
            {
                // Fix constrained voxels by updating the initial deformation field
                constrain_deformation_field(
                    df,
                    _constraints_mask_pyramid.volume(l),
                    _constraints_pyramid.volume(l)
                );
            }

            UnaryFunction unary_fn;
            build_unary_function(l, unary_fn);

            if (_settings.solver == Settings::Solver_ICM) {
                BlockedGraphCutOptimizer<ICMSolver<double>> optimizer(
                    neighborhood,
                    _settings.levels[l].block_size,
                    _settings.levels[l].block_energy_epsilon,
                    _settings.levels[l].max_iteration_count
                );
                optimizer.execute(
                    unary_fn,
                    binary_fn,
                    _settings.levels[l].step_size,
                    _settings.update_rule,
                    df
                );
            }
#if defined(DF_ENABLE_GCO)
            else if (_settings.solver == Settings::Solver_GCO) {
                BlockedGraphCutOptimizer<GCOSolver<double>> optimizer(
                    neighborhood,
                    _settings.levels[l].block_size,
                    _settings.levels[l].block_energy_epsilon,
                    _settings.levels[l].max_iteration_count
                );
                optimizer.execute(
                    unary_fn,
                    binary_fn,
                    _settings.levels[l].step_size,
                    _settings.update_rule,
                    df
                );
            }
#endif
#if defined(DF_ENABLE_GRIDCUT)
            else if (_settings.solver == Settings::Solver_GridCut) {
                BlockedGraphCutOptimizer<GridCutSolver<double>> optimizer(
                    neighborhood,
                    _settings.levels[l].block_size,
                    _settings.levels[l].block_energy_epsilon,
                    _settings.levels[l].max_iteration_count
                );
                optimizer.execute(
                    unary_fn,
                    binary_fn,
                    _settings.levels[l].step_size,
                    _settings.update_rule,
                    df
                );
            }
#endif

        }
        else {
            LOG(Info) << "Skipping level " << l;
        }

        stk::VolumeFloat3 vf = df.volume();
        if (l != 0) {

            dim3 upsampled_dims = _deformation_pyramid.volume(l - 1).size();
            _deformation_pyramid.set_volume(l - 1,
            #ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
                filters::upsample_vectorfield(vf, upsampled_dims, _deformation_pyramid.residual(l - 1))
            #else
                filters::upsample_vectorfield(vf, upsampled_dims)
            #endif
            );

            #ifdef DF_OUTPUT_DEBUG_VOLUMES
                upsample_and_save(l);
            #endif // DF_OUTPUT_DEBUG_VOLUMES
        }
        else {
            _deformation_pyramid.set_volume(0, vf);
        }
    }

    return _deformation_pyramid.volume(0);
}
stk::Volume RegistrationEngine::deformation_field(int level)
{
    return _deformation_pyramid.volume(level);
}
std::vector<int3> RegistrationEngine::determine_neighborhood(int level) const
{
    /*
        To enable registration of images with lower dimensionality than 3 we
        automatically adopt the neighborhood for the optimization. Given a set
        of images (all fixed and moving images) we determine the dimensionality
        for the registration by assuming that if all images are of size 1 in a
        specific dimension, that dimension is not in play.

        E.g., given an input of only WxHx1 images, we trigger 2D registration
        by setting the search neighborhood to only the X and Y axes.
    */

    std::vector<int3> neighborhood;

    dim3 dim_size {0, 0, 0};

    for (int i = 0; i < (int) _fixed_pyramids.size(); ++i) {
        stk::Volume fixed;
        stk::Volume moving;

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

#ifdef DF_OUTPUT_DEBUG_VOLUMES
void RegistrationEngine::upsample_and_save(int level)
{
    if (level == 0) return;

    int target_level = 0;
    int diff = level - target_level;
    assert(diff > 0);

    stk::VolumeHelper<float3> def = _deformation_pyramid.volume(target_level);
    stk::VolumeHelper<float3> def_low = _deformation_pyramid.volume(level);

    dim3 dims = def.size();

    float factor = powf(0.5f, float(diff));

    #pragma omp parallel for
    for (int z = 0; z < int(dims.z); ++z) {
        for (int y = 0; y < int(dims.y); ++y) {
            for (int x = 0; x < int(dims.x); ++x) {
                def(x, y, z) = (1.0f/factor) * def_low.linear_at(factor*x, factor*y, factor*z, volume::Border_Replicate);
            }
        }
    }

    std::stringstream ss;
    ss << "deformation_l" << level << ".vtk";
    std::string tmp = ss.str();
    stk::write_volume(tmp.c_str(), def);

    ss.str("");
    ss << "deformation_low_l" << level << ".vtk";
    tmp = ss.str();
    stk::write_volume(tmp.c_str(), def_low);

    stk::Volume moving = _moving_pyramids[0].volume(0);

    ss.str("");
    ss << "transformed_l" << level << ".vtk";
    tmp = ss.str();
    stk::write_volume(tmp.c_str(), transform_volume(moving, def));

    ss.str("");
    ss << "transformed_low_l" << level << ".vtk";
    tmp = ss.str();
    stk::write_volume(tmp.c_str(), transform_volume(moving, def_low));
}
void RegistrationEngine::save_volume_pyramid()
{
    ASSERT(_fixed_pyramids.size() == _moving_pyramids.size());
    for (int l = 0; l < _settings.num_pyramid_levels; ++l) {
        for (int i = 0; i < _fixed_pyramids.size(); ++i) {

            if (_fixed_pyramids[i].levels() > 0) {
                std::stringstream file;
                file << "fixed_pyramid_" << i << "_level_" << l << ".vtk";
                stk::write_volume(file.str().c_str(), _fixed_pyramids[i].volume(l));
            }

            if (_moving_pyramids[i].levels() > 0) {
                std::stringstream file;
                file << "moving_pyramid_" << i << "_level_" << l << ".vtk";
                stk::write_volume(file.str().c_str(), _moving_pyramids[i].volume(l));
            }
        }

        std::stringstream file;
        file << "initial_deformation_level_" << l << ".vtk";
        stk::write_volume(file.str().c_str(), _deformation_pyramid.volume(l));
    }
}
#endif // DF_OUTPUT_DEBUG_VOLUMES
