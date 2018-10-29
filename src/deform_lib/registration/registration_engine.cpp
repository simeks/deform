#include "../config.h"
#include "../cost_functions/cost_function.h"
#include "../filters/resample.h"

#include "blocked_graph_cut_optimizer.h"
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

    void constrain_deformation_field(stk::VolumeFloat3& def,
        const stk::VolumeUChar& mask, const stk::VolumeFloat3& values)
    {
        ASSERT(def.size() == mask.size() && def.size() == values.size());
        dim3 dims = def.size();
        for (int z = 0; z < int(dims.z); ++z) {
            for (int y = 0; y < int(dims.y); ++y) {
                for (int x = 0; x < int(dims.x); ++x) {
                    if (mask(x, y, z) > 0) {
                        def(x, y, z) = values(x, y, z);
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

        std::unique_ptr<SubFunction> function = std::make_unique<SquaredDistanceFunction<T>>(fixed, moving);
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

        for (const auto& [k, v] : parameters) {
            if (k == "radius") {
                radius = str_to_num<int>("NCCFunction", k, v);
            }
            else if (k == "window") {
                if (v != "cube" && v != "sphere") {
                    throw std::invalid_argument("NCCFunction: invalid value '" + v +
                                                "' for parameter '" + k + "'");
                }
                window = v;
            }
            else {
                throw std::invalid_argument("NCCFunction: unrecognised parameter "
                                            "'" + k + "' with value '" + v + "'");
            }
        }

        std::unique_ptr<SubFunction> function;
        if ("sphere" == window) {
            function = std::make_unique<NCCFunction_sphere<T>>(fixed, moving, radius);
            if (moving_mask.valid()) {
                function->set_moving_mask(moving_mask);
            }
        }
        else if ("cube" == window) {
            function = std::make_unique<NCCFunction_cube<T>>(fixed, moving, radius);
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

        for (const auto& [k, v] : parameters) {
            if (k == "bins") {
                bins = str_to_num<int>("MIFunction", k, v);
            }
            else if (k == "sigma") {
                sigma = str_to_num<double>("MIFunction", k, v);
            }
            else if (k == "update_interval") {
                update_interval = str_to_num<int>("MIFunction", k, v);
            }
            else if (k == "interpolator") {
                if (v == "linear") {
                    interpolator = transform::Interp_Linear;
                }
                else if (v == "nearest" || v == "nn") {
                    interpolator = transform::Interp_NN;
                }
                else {
                    throw std::invalid_argument("MIFunction: invalid interpolator '" + v + "'");
                }
            }
            else {
                throw std::invalid_argument("MIFunction: unrecognised parameter "
                                            "'" + k + "' with value '" + v + "'");
            }
        }

        std::unique_ptr<SubFunction> function =
            std::make_unique<MIFunction<T>>(fixed, moving, bins, sigma, update_interval, interpolator);
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

        for (const auto& [k, v] : parameters) {
            if (k == "sigma") {
                sigma = str_to_num<float>("GradientSSDFunction", k, v);
            }
            else {
                throw std::invalid_argument("GradientSSDFunction: unrecognised parameter "
                                            "'" + k + "' with value '" + v + "'");
            }
        }

        std::unique_ptr<SubFunction> function =
            std::make_unique<GradientSSDFunction<T>>(fixed, moving, sigma);
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

    // Clone the def, because the current copy will be changed when executing the optimizer
    binary_fn.set_initial_displacement(_deformation_pyramid.volume(level).clone());

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
    
    #ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        if (_regularization_weight_map.volume(level).valid())
            binary_fn.set_weight_map(_regularization_weight_map.volume(l));
    #endif

    auto const& moving_mask = _moving_mask_pyramid.levels() > 0 ? _moving_mask_pyramid.volume(level)
                                                                : stk::VolumeFloat();

    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
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
            std::make_unique<LandmarksFunction>(
                _fixed_landmarks,
                _moving_landmarks,
                fixed
            ),
            _settings.levels[level].landmarks_weight
        );
    }

    if (_constraints_mask_pyramid.volume(level).valid()) {
        unary_fn.add_function(
            std::make_unique<SoftConstraintsFunction>(
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
    _fixed_pyramids.resize(DF_MAX_IMAGE_PAIR_COUNT);
    _moving_pyramids.resize(DF_MAX_IMAGE_PAIR_COUNT);

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
    ASSERT(i < DF_MAX_IMAGE_PAIR_COUNT);

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
        stk::VolumeFloat3 def = _deformation_pyramid.volume(l);

        if (l >= _settings.pyramid_stop_level) {
            LOG(Info) << "Performing registration level " << l;

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

            UnaryFunction unary_fn;
            build_unary_function(l, unary_fn);

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
            #ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
                filters::upsample_vectorfield(def, upsampled_dims, _deformation_pyramid.residual(l - 1))
            #else
                filters::upsample_vectorfield(def, upsampled_dims)
            #endif
            );

            #ifdef DF_OUTPUT_DEBUG_VOLUMES
                upsample_and_save(l);
            #endif // DF_OUTPUT_DEBUG_VOLUMES
        }
        else {
            _deformation_pyramid.set_volume(0, def);
        }
    }

    return _deformation_pyramid.volume(0);
}
stk::Volume RegistrationEngine::deformation_field(int level)
{
    return _deformation_pyramid.volume(level);
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
    for (int l = 0; l < _settings.num_pyramid_levels; ++l) {
        for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {

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
