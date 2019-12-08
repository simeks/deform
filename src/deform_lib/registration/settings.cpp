#include "settings.h"

#include <stk/common/log.h>
#include <stk/math/float3.h>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4127)
#endif

#include <yaml-cpp/yaml.h>

#ifdef _MSC_VER
    #pragma warning(pop)
#endif


#include <fstream>

/*
pyramid_levels: 6
pyramid_stop_level: 0
field_downscale_factor: 1

solver: gco
update_rule: additive

constraints_weight: 1000

block_size: [12, 12, 12]
block_energy_epsilon: 0.001
step_size: 0.5
regularization_weight: 0.05
regularization_scale: 1.0
regularization_exponent: 2.0

levels:
    0:
        regularization_weight: 0
        step_size: 0.1
    1:
        regularization_weight: 1
        step_size: 0.1

image_slots:

  # water
  - resampler: gaussian
    normalize: true
    cost_function:
      - function: ssd
        weight: 0.3
      - function: ncc
        weight: 0.4

  # fat
  - resampler: gaussian
    normalize: true
    cost_function: squared_distance

  # sfcm
  - resampler: gaussian
    normalize: true
    cost_function: squared_distance

*/

namespace YAML {

    template<>
    struct convert<int3>
    {
        static bool decode(const Node& node, int3& out) {
            if(!node.IsSequence() || node.size() != 3) {
                throw YAML::RepresentationException(node.Mark(), "expected vector of 3 integers");
            }

            try {
                out = {
                    node[0].as<int>(),
                    node[1].as<int>(),
                    node[2].as<int>()
                };
            }
            catch (YAML::TypedBadConversion<int>&) {
                throw YAML::RepresentationException(node.Mark(), "expected vector of 3 integers");
            }

            return true;
        }
    };

    template<>
    struct convert<float3>
    {
        static bool decode(const Node& node, float3& out) {
            if(!node.IsSequence() || node.size() != 3) {
                throw YAML::RepresentationException(node.Mark(), "expected vector of 3 floats");
            }

            try {
                out = {
                    node[0].as<float>(),
                    node[1].as<float>(),
                    node[2].as<float>()
                };
            }
            catch (YAML::TypedBadConversion<float>&) {
                throw YAML::RepresentationException(node.Mark(), "expected vector of 3 floats");
            }

            return true;
        }
    };

    template<>
    struct convert<Settings::ImageSlot::ResampleMethod>
    {
        static bool decode(const Node& node, Settings::ImageSlot::ResampleMethod& out) {
            if(!node.IsScalar()) {
                throw YAML::RepresentationException(node.Mark(), "expected resampler");
            }

            std::string fn;
            try {
                fn = node.as<std::string>();
            }
            catch (YAML::TypedBadConversion<std::string> &) {
                throw YAML::RepresentationException(node.Mark(), "expected resampler");
            }

            if (fn == "gaussian") {
                out = Settings::ImageSlot::Resample_Gaussian;
                return true;
            }

            throw YAML::RepresentationException(node.Mark(), "unrecognised resampler " + fn);
        }
    };

    template<>
    struct convert<Settings::ImageSlot::CostFunction>
    {
        static bool decode(const Node& node, Settings::ImageSlot::CostFunction& out) {
            if(!node.IsScalar()) {
                throw YAML::RepresentationException(node.Mark(), "expected cost function");
            }

            std::string fn;
            try {
                fn = node.as<std::string>();
            }
            catch (YAML::TypedBadConversion<std::string> &) {
                throw YAML::RepresentationException(node.Mark(), "expected cost function");
            }

            if (fn == "none") {
                out = Settings::ImageSlot::CostFunction_None;
                return true;
            }
            else if (fn == "squared_distance" || fn == "ssd") {
                out = Settings::ImageSlot::CostFunction_SSD;
                return true;
            }
            else if (fn == "ncc") {
                out = Settings::ImageSlot::CostFunction_NCC;
                return true;
            }
            else if (fn == "mutual_information" || fn == "mi") {
                out = Settings::ImageSlot::CostFunction_MI;
                return true;
            }
            else if (fn == "gradient_ssd") {
                out = Settings::ImageSlot::CostFunction_Gradient_SSD;
                return true;
            }

            throw YAML::RepresentationException(node.Mark(), "unrecognised cost function " + fn);
        }
    };

    template<>
    struct convert<Settings::ImageSlot::WeightedFunction>
    {
        static bool decode(const Node& node, Settings::ImageSlot::WeightedFunction& out) {
            if(!node.IsMap() || !node["function"]) {
                throw YAML::RepresentationException(node.Mark(), "expected cost function");
            }

            out.function = node["function"].as<Settings::ImageSlot::CostFunction>();

            // Optional parameter weight
            if (node["weight"]) {
                out.weight = node["weight"].as<float>();
            }
            else {
                out.weight = 1.0f;
            }

            // Function parameters
            for(auto it = node.begin(); it != node.end(); ++it) {
                auto k = it->first.as<std::string>();
                if (k == "function" || k == "weight") {
                    continue;
                }
                out.parameters.emplace(k, it->second.as<std::string>());
            }

            return true;
        }
    };

    template<>
    struct convert<Settings::ImageSlot>
    {
        static bool decode(const Node& node, Settings::ImageSlot& out) {
            if(!node.IsMap()) {
                throw YAML::RepresentationException(node.Mark(), "expected image slot");
            }

            for (const auto& c : node) {
                std::string key = c.first.as<std::string>();
                const YAML::Node& value = c.second;

                if (key == "cost_function") {
                    if (value.IsSequence()) {
                        out.cost_functions.resize(value.size());
                        for(size_t k = 0; k < value.size(); ++k) {
                            out.cost_functions[k] = value[k].as<Settings::ImageSlot::WeightedFunction>();
                        }
                    }
                    else {
                        // NOTE: assuming that the constructor of ImageSlot initialises
                        // at least the first cost function in the array
                        out.cost_functions[0].function = value.as<Settings::ImageSlot::CostFunction>();
                        out.cost_functions[0].weight = 1.0f;
                    }
                }
                else if (key == "resampler") {
                    out.resample_method = value.as<Settings::ImageSlot::ResampleMethod>();
                }
                else if (key == "normalize") {
                    out.normalize = value.as<bool>();
                }
                else {
                    throw YAML::RepresentationException(node.Mark(), "Unrecognized image slot parameter: " + key);
                }
            }
            return true;
        }
    };

} // namespace YAML

const char* cost_function_to_str(Settings::ImageSlot::CostFunction fn)
{
    switch (fn) {
    case Settings::ImageSlot::CostFunction_SSD:
        return "ssd";
    case Settings::ImageSlot::CostFunction_NCC:
        return "ncc";
    case Settings::ImageSlot::CostFunction_MI:
        return "mi";
    case Settings::ImageSlot::CostFunction_Gradient_SSD:
        return "gradient_ssd";
    default:
    case Settings::ImageSlot::CostFunction_None:
        return "none";
    }
}
const char* resample_method_to_str(const Settings::ImageSlot::ResampleMethod fn)
{
    switch (fn) {
    case Settings::ImageSlot::Resample_Gaussian:
        return "gaussian";
    };
    return "none";
}
const char* solver_to_str(Settings::Solver solver)
{
    switch (solver) {
    case Settings::Solver_ICM: return "icm";
#ifdef DF_ENABLE_GCO
    case Settings::Solver_GCO: return "gco";
#endif
#ifdef DF_ENABLE_GRIDCUT
    case Settings::Solver_GridCut: return "gridcut";
#endif
    };
    return "none";
}
const char* update_rule_to_str(Settings::UpdateRule op)
{
    switch (op) {
    case Settings::UpdateRule_Additive: return "additive";
    case Settings::UpdateRule_Compositive: return "compositive";
    };
    return "none";
}

// Parses a level specific parameter
// Returns true if any parameter was actually set, false if not
static bool parse_level_parameter(
    const std::string& key,
    const YAML::Node& value,
    Settings::Level& out
)
{
    if (key == "block_size") {
        out.block_size = value.as<int3>();
    }
    else if (key == "block_energy_epsilon") {
        out.block_energy_epsilon = value.as<float>();
    }
    else if (key == "max_iteration_count") {
        out.max_iteration_count = value.as<int>();
    }
    else if (key == "regularization_weight") {
        out.regularization_weight = value.as<float>();
    }
    else if (key == "regularization_scale") {
        out.regularization_scale = value.as<float>();
    }
    else if (key == "regularization_exponent") {
        out.regularization_exponent = value.as<float>();
    }
    else if (key == "step_size") {
        try {
            out.step_size = value.as<float3>();
        }
        catch (YAML::RepresentationException&) {
            try {
                float f = value.as<float>();
                out.step_size = {f, f, f};
            }
            catch (YAML::RepresentationException&) {
                throw YAML::RepresentationException(
                        value.Mark(),
                        "expected float or sequence of three floats"
                        );
            }
        }
        
        float len = stk::norm(out.step_size);
        if (len == 0 ||
            out.step_size.x < 0 ||
            out.step_size.y < 0 ||
            out.step_size.z < 0) {
            throw ValidationError("Settings: Invalid step_size, step_size should be greater than zero");
        }
    }
    else if (key == "constraints_weight") {
        out.constraints_weight = value.as<float>();
    }
    else if (key == "landmarks_weight") {
        out.landmarks_weight = value.as<float>();
    }
    else if (key == "landmarks_decay") {
        out.landmarks_decay = value.as<float>();
    }
    else {
        return false;
    }
    return true;
}


void print_registration_settings(const Settings& settings, std::ostream& s)
{
    s << "Settings:" << std::endl;
    s << "pyramid_stop_level = " << settings.pyramid_stop_level << std::endl;
    s << "num_pyramid_levels = " << settings.num_pyramid_levels << std::endl;
    s << "field_downscale_factor = " << settings.field_downscale_factor << std::endl;
    s << "landmarks_stop_level = " << settings.landmarks_stop_level << std::endl;
    s << "solver = " << solver_to_str(settings.solver) << std::endl;
    s << "update_rule = " << update_rule_to_str(settings.update_rule) << std::endl;

    for (int l = 0; l < settings.num_pyramid_levels; ++l) {
        s << "level[" << l << "] = {" << std::endl;
        s << "  block_size = " << settings.levels[l].block_size << std::endl;
        s << "  block_energy_epsilon = " << settings.levels[l].block_energy_epsilon << std::endl;
        s << "  max_iteration_count = " << settings.levels[l].max_iteration_count << std::endl;
        s << "  step_size = " << settings.levels[l].step_size << std::endl;
        s << "  regularization_weight = " << settings.levels[l].regularization_weight << std::endl;
        s << "  constraints_weight = " << settings.levels[l].constraints_weight << std::endl;
        s << "  regularization_scale = " << settings.levels[l].regularization_scale << std::endl;
        s << "  regularization_exponent = " << settings.levels[l].regularization_exponent << std::endl;
        s << "  landmarks_weight = " << settings.levels[l].landmarks_weight << std::endl;
        s << "  landmarks_decay = " << settings.levels[l].landmarks_decay << std::endl;
        s << "}" << std::endl;
    }

    for (int i = 0; i < (int) settings.image_slots.size(); ++i) {
        auto slot = settings.image_slots[i];

        // Dont print unused slots
        if (0 == slot.cost_functions.size() ||
                Settings::ImageSlot::CostFunction_None == slot.cost_functions[0].function) {
            continue;
        }

        s << "image_slot[" << i << "] = {" << std::endl;
        s << "  resample_method = " << resample_method_to_str(slot.resample_method) << std::endl;
        s << "  normalize = " << (slot.normalize ? "true" : "false") << std::endl;
        s << "  cost_functions = {" << std::endl;
        for (size_t j = 0; j < slot.cost_functions.size(); ++j) {
            s << "    " << cost_function_to_str(slot.cost_functions[j].function) << ": " << std::endl;
            s << "      weight: " << slot.cost_functions[j].weight << std::endl;
            for (const auto& p : slot.cost_functions[j].parameters) {
                s << "      " << p.first << ": " << p.second << std::endl;
            }
        }
        s << "  }" << std::endl;
        s << "}" << std::endl;
    }
}

void parse_registration_settings(const std::string& str, Settings& settings)
{
    settings = {}; // Clean up

    try {
    
    YAML::Node root = YAML::Load(str);

    // First pass we parse global level settings
    Settings::Level global_level_settings;

    // Global settings not connected to specific levels
    for (const auto& node : root) {
        std::string key = node.first.as<std::string>();
        const YAML::Node& value = node.second;
        
        if (key == "pyramid_levels") {
            settings.num_pyramid_levels = value.as<int>();
        }
        else if (key == "pyramid_stop_level") {
            settings.pyramid_stop_level = value.as<int>();
        }
        else if (key == "field_downscale_factor") {
            settings.field_downscale_factor = value.as<int>();
        }
        else if (key == "landmarks_stop_level") {
            settings.landmarks_stop_level = value.as<int>();
        }
        else if (key == "regularize_initial_displacement") {
            LOG(Warning) << "'regularize_initial_displacement' is no longer supported";
        }
        else if (key == "solver") {
            std::string solver = value.as<std::string>();

            if (solver == "icm") {
                settings.solver = Settings::Solver_ICM;
            }
            else if (solver == "gco") {
            #ifdef DF_ENABLE_GCO
                settings.solver = Settings::Solver_GCO;
            #else
                throw ValidationError("Settings: Solver 'gco' not enabled");
            #endif
            }
            else if (solver == "gridcut") {
            #ifdef DF_ENABLE_GRIDCUT
                settings.solver = Settings::Solver_GridCut;
            #else
                throw ValidationError("Settings: Solver 'gridcut' not enabled");
            #endif
            }
            else {
                throw ValidationError("Settings: Invalid solver");
            }

        }
        else if (key == "update_rule") {
            std::string rule = value.as<std::string>();
            if (rule == "additive") {
                settings.update_rule = Settings::UpdateRule_Additive;
            }
            else if (rule == "compositive") {
                settings.update_rule = Settings::UpdateRule_Compositive;

                // We have only proven that the compositive energy function is submodular
                // for regularization_exponent=2
                for (int i = settings.pyramid_stop_level; i < settings.num_pyramid_levels; ++i) {
                    if (settings.levels[i].regularization_exponent != 2) {
                        LOG(Warning) << "Submodularity is only guaranteed for "
                                        << "regularization_exponent=2 when using the "
                                        << "compositive update rule";
                        break;
                    }
                }
            }
            else {
                throw ValidationError("Settings: Invalid update rule");
            }
        }
        else if (parse_level_parameter(key, value, global_level_settings)) {
            // parse_level_parameter does the parsing
        }
        else if (key == "image_slots") {
            if (value.IsSequence()) {
                for (size_t i = 0; i < value.size(); ++i) {
                    settings.image_slots.push_back(value[i].as<Settings::ImageSlot>());
                }
            } else {
                throw ValidationError("Settings: Expeced 'image_slots' to be a sequence");
            }
        }
        else if (key == "levels") {
            // We parse levels in a second pass, to allow global settings to be set
        }
        else {
            std::stringstream ss;
            ss << "Settings: Unrecognized parameter: " << key;
            throw ValidationError(ss.str());
        }
    }

    // Apply global settings for all levels
    settings.levels.resize(settings.num_pyramid_levels);
    for (int i = 0; i < settings.num_pyramid_levels; ++i) {
        settings.levels[i] = global_level_settings;
    }

    // Parse per-level overrides
    auto levels = root["levels"];
    if (levels) {
        for (const auto& level : levels) {
            int l = level.first.as<int>();
            if (l >= settings.num_pyramid_levels) {
                throw ValidationError("Settings: index of level exceed number specified in pyramid_levels");
            }

            if(!level.second.IsMap()) {
                throw YAML::RepresentationException(level.second.Mark(), "expected level");
            }

            for (const auto& node : level.second) {
                std::string key = node.first.as<std::string>();
                if (!parse_level_parameter(key, node.second, settings.levels[l])) {
                    std::stringstream ss;
                    ss << "Settings: Unrecognized level parameter: " << node.first.as<std::string>();
                    throw ValidationError(ss.str());
                }
            }
        }
    }

    }
    catch (YAML::Exception& e) {
        std::stringstream ss;
        ss << "Settings: " << e.what();
        
        throw ValidationError(ss.str());
    }
}
