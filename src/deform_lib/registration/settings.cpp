#include "settings.h"

#include <stk/common/log.h>

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

            // Cost functions
            auto& cf = node["cost_function"];
            if (cf) {
                if (cf.IsSequence()) {
                    out.cost_functions.resize(cf.size());
                    for(size_t k = 0; k < cf.size(); ++k) {
                        out.cost_functions[k] = cf[k].as<Settings::ImageSlot::WeightedFunction>();
                    }
                }
                else {
                    // NOTE: assuming that the constructor of ImageSlot initialises
                    // at least the first cost function in the array
                    out.cost_functions[0].function = cf.as<Settings::ImageSlot::CostFunction>();
                    out.cost_functions[0].weight = 1.0f;
                }
            }

            // Resampling method
            if (node["resampler"]) {
                out.resample_method = node["resampler"].as<Settings::ImageSlot::ResampleMethod>();
            }

            // Normalisation
            if (node["normalize"]) {
                try {
                    out.normalize = node["normalize"].as<bool>();
                }
                catch (YAML::TypedBadConversion<bool> &) {
                    throw YAML::RepresentationException(node.Mark(), "expected bool");
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

static void parse_level(const YAML::Node& node, Settings::Level& out) {
    if(!node.IsMap()) {
        throw YAML::RepresentationException(node.Mark(), "expected level");
    }

    if (node["block_size"]) {
        out.block_size = node["block_size"].as<int3>();
    }

    if (node["block_energy_epsilon"]) {
        out.block_energy_epsilon = node["block_energy_epsilon"].as<float>();
    }

    if (node["max_iteration_count"]) {
        out.max_iteration_count = node["max_iteration_count"].as<int>();
    }

    if (node["regularization_weight"]) {
        out.regularization_weight = node["regularization_weight"].as<float>();
    }

    if (node["regularization_scale"]) {
        out.regularization_scale = node["regularization_scale"].as<float>();
    }

    if (node["regularization_exponent"]) {
        out.regularization_exponent = node["regularization_exponent"].as<float>();
    }

    if (node["step_size"]) {
        try {
            out.step_size = node["step_size"].as<float3>();
        }
        catch (YAML::RepresentationException&) {
            try {
                float f = node["step_size"].as<float>();
                out.step_size = {f, f, f};
            }
            catch (YAML::RepresentationException&) {
                throw YAML::RepresentationException(
                        node["step_size"].Mark(),
                        "expected float or sequence of three floats"
                        );
            }
        }
    }

    if (node["constraints_weight"]) {
        out.constraints_weight = node["constraints_weight"].as<float>();
    }

    if (node["landmarks_weight"]) {
        out.landmarks_weight = node["landmarks_weight"].as<float>();
    }

    if (node["landmarks_decay"]) {
        out.landmarks_decay = node["landmarks_decay"].as<float>();
    }
}


void print_registration_settings(const Settings& settings, std::ostream& s)
{
    s << "Settings:";
    s << "pyramid_stop_level = " << settings.pyramid_stop_level;
    s << "num_pyramid_levels = " << settings.num_pyramid_levels;
    s << "landmarks_stop_level = " << settings.landmarks_stop_level;

    for (int l = 0; l < settings.num_pyramid_levels; ++l) {
        s << "level[" << l << "] = {";
        s << "  block_size = " << settings.levels[l].block_size;
        s << "  block_energy_epsilon = " << settings.levels[l].block_energy_epsilon;
        s << "  max_iteration_count = " << settings.levels[l].max_iteration_count;
        s << "  step_size = " << settings.levels[l].step_size;
        s << "  regularization_weight = " << settings.levels[l].regularization_weight;
        s << "  constraints_weight = " << settings.levels[l].constraints_weight;
        s << "  regularization_scale = " << settings.levels[l].regularization_scale;
        s << "  regularization_exponent = " << settings.levels[l].regularization_exponent;
        s << "  landmarks_weight = " << settings.levels[l].landmarks_weight;
        s << "  landmarks_decay = " << settings.levels[l].landmarks_decay;
        s << "}";
    }

    for (int i = 0; i < (int) settings.image_slots.size(); ++i) {
        auto slot = settings.image_slots[i];

        // Dont print unused slots
        if (0 == slot.cost_functions.size() ||
                Settings::ImageSlot::CostFunction_None == slot.cost_functions[0].function) {
            continue;
        }

        s << "image_slot[" << i << "] = {";
        s << "  resample_method = " << resample_method_to_str(slot.resample_method);
        s << "  normalize = " << (slot.normalize ? "true" : "false");
        s << "  cost_functions = {";
        for (size_t j = 0; j < slot.cost_functions.size(); ++j) {
            s << "    " << cost_function_to_str(slot.cost_functions[j].function) << ": ";
            s << "      weight: " << slot.cost_functions[j].weight;
            for (const auto& [k, v] : slot.cost_functions[j].parameters) {
                s << "      " << k << ": " << v;
            }
        }
        s << "  }";
        s << "}";
    }
}


bool parse_registration_settings(const std::string& str, Settings& settings)
{
    settings = {}; // Clean up

    try {

        YAML::Node root = YAML::Load(str);

        if (root["pyramid_levels"]) {
            settings.num_pyramid_levels = root["pyramid_levels"].as<int>();
        }

        if (root["pyramid_stop_level"]) {
            settings.pyramid_stop_level = root["pyramid_stop_level"].as<int>();
        }

        // First parse global level settings
        Settings::Level global_level_settings;
        parse_level(root, global_level_settings);

        // Apply global settings for all levels
        settings.levels.resize(settings.num_pyramid_levels);
        for (int i = 0; i < settings.num_pyramid_levels; ++i) {
            settings.levels[i] = global_level_settings;
        }

        // Parse per-level overrides
        auto levels = root["levels"];
        if (levels) {
            for (auto it = levels.begin(); it != levels.end(); ++it) {
                int l = it->first.as<int>();
                if (l >= settings.num_pyramid_levels) {
                    throw ValidationError("Settings: index of level exceed number specified in pyramid_levels");
                }
                parse_level(it->second, settings.levels[l]);
            }
        }

        if (root["landmarks_stop_level"]) {
            settings.landmarks_stop_level = root["landmarks_stop_level"].as<int>();
        }

        auto is = root["image_slots"];
        if (is && is.IsSequence()) {
            for (size_t i = 0; i < is.size(); ++i) {
                settings.image_slots.push_back(is[i].as<Settings::ImageSlot>());
            }
        }
    }
    catch (YAML::Exception& e) {
        LOG(Error) << "[Settings] " << e.what();
        return false;
    }

    return true;
}

bool parse_registration_file(const std::string& parameter_file, Settings& settings)
{
    // Defaults
    settings = Settings();

    std::ifstream f(parameter_file, std::ifstream::in);
    if (!f.is_open()) {
        LOG(Error) << "[Settings] Failed to open file '" << parameter_file << "'";
        return false;
    }

    std::stringstream ss;
    ss << f.rdbuf();

    return parse_registration_settings(ss.str(), settings);
}

