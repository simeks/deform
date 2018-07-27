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
    default:
    case Settings::ImageSlot::CostFunction_None:
        return "none";
    }
}
const char* resample_method_to_str(Settings::ImageSlot::ResampleMethod fn)
{
    switch (fn) {
    case Settings::ImageSlot::Resample_Gaussian:
        return "gaussian";
    };
    return "none";
}

void print_registration_settings(const Settings& settings)
{
    LOG(Info) << "Settings:";
    LOG(Info) << "pyramid_stop_level = " << settings.pyramid_stop_level;
    LOG(Info) << "num_pyramid_levels = " << settings.num_pyramid_levels;
    LOG(Info) << "block_size = " << settings.block_size; 
    LOG(Info) << "block_energy_epsilon = " << settings.block_energy_epsilon;
    LOG(Info) << "step_size = " << settings.step_size;
    LOG(Info) << "regularization_weight = " << settings.regularization_weight;
    LOG(Info) << "landmarks_weight = " << settings.landmarks_weight;
    LOG(Info) << "landmarks_stop_level = " << settings.landmarks_stop_level;
    LOG(Info) << "constraints_weight = " << settings.constraints_weight;

    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
        auto slot = settings.image_slots[i];

        // Dont print unused slots
        if (0 == slot.cost_functions.size() || 
                Settings::ImageSlot::CostFunction_None == slot.cost_functions[0].function) {
            continue;
        }

        LOG(Info) << "image_slot[" << i << "] = {";
        LOG(Info) << "  resample_method = " << resample_method_to_str(slot.resample_method);        
        LOG(Info) << "  normalize = " << (slot.normalize ? "true" : "false");        
        LOG(Info) << "  cost_functions = {";
        for (size_t k = 0; k < slot.cost_functions.size(); ++k) {
            LOG(Info) << "    " << cost_function_to_str(slot.cost_functions[k].function) << ": "
                                << slot.cost_functions[k].weight;
        }
        LOG(Info) << "  }";
        LOG(Info) << "}";
    }
}


bool parse_registration_settings(const std::string& str, Settings& settings)
{
    try {

        YAML::Node root = YAML::Load(str);
        
        if (root["pyramid_levels"]) {
            settings.num_pyramid_levels = root["pyramid_levels"].as<int>();
        }

        if (root["pyramid_stop_level"]) {
            settings.pyramid_stop_level = root["pyramid_stop_level"].as<int>();
        }

        if (root["step_size"]) {
            settings.step_size = root["step_size"].as<float>();
        }

        if (root["regularization_weight"]) {
            settings.regularization_weight = root["regularization_weight"].as<float>();
        }

        if (root["block_size"]) {
            settings.block_size = root["block_size"].as<int3>();
        }

        if (root["block_energy_epsilon"]) {
            settings.block_energy_epsilon = root["block_energy_epsilon"].as<float>();
        }

        if (root["constraints_weight"]) {
            settings.constraints_weight = root["constraints_weight"].as<float>();
        }

        if (root["landmarks_weight"]) {
            settings.landmarks_weight = root["landmarks_weight"].as<float>();
        }

        if (root["landmarks_stop_level"]) {
            settings.landmarks_stop_level = root["landmarks_stop_level"].as<int>();
        }

        auto is = root["image_slots"];
        if (is && is.IsSequence()) {
            for (size_t i = 0; i < is.size() && i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
                settings.image_slots[i] = is[i].as<Settings::ImageSlot>();
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

