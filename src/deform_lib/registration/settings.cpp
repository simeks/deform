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

class UnrecognisedCostFunction : public YAML::RepresentationException {
public:
    UnrecognisedCostFunction(const YAML::Mark& mark_)
      : RepresentationException(mark_, "invalid cost function") {}
    UnrecognisedCostFunction(const UnrecognisedCostFunction&) = default;
    virtual ~UnrecognisedCostFunction() noexcept {};
};

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


// Return true on success, false on failure
template<typename T>
bool read_value(const YAML::Node& obj, const char* name, T& out);

template<>
bool read_value<int>(const YAML::Node& obj, const char* name, int& out)
{
    try {
        out = obj[name].as<int>();
    }
    catch (YAML::TypedBadConversion<int>&) {
        LOG(Error) << "Settings: '" << name << "', expected integer";
        return false;
    }
    return true;
}
template<>
bool read_value<float>(const YAML::Node& obj, const char* name, float& out)
{
    try {
        out = obj[name].as<float>();
    }
    catch (YAML::TypedBadConversion<float>&) {
        LOG(Error) << "Settings: '" << name << "', expected float";
        return false;
    }
    return true;
}
template<>
bool read_value<double>(const YAML::Node& obj, const char* name, double& out)
{
    try {
        out = obj[name].as<double>();
    }
    catch (YAML::TypedBadConversion<double>&) {
        LOG(Error) << "Settings: '" << name << "', expected double";
        return false;
    }
    return true;
}
template<>
bool read_value<bool>(const YAML::Node& obj, const char* name, bool& out)
{
    try {
        out = obj[name].as<bool>();
    }
    catch (YAML::TypedBadConversion<bool>&) {
        LOG(Error) << "Settings: '" << name << "', expected boolean";
        return false;
    }
    return true;
}
template<>
bool read_value<Settings::ImageSlot::ResampleMethod>(const YAML::Node& obj, 
    const char* name, Settings::ImageSlot::ResampleMethod& out)
{
    std::string fn;
    try {
        fn = obj[name].as<std::string>();
    }
    catch (YAML::TypedBadConversion<std::string>&) {
        LOG(Error) << "Settings: '" << name << "', expected string";
        return false;
    }

    if (fn == "gaussian") {
        out = Settings::ImageSlot::Resample_Gaussian;
    }
    else {
        LOG(Error) << "Settings: Unrecognized value '" << fn << "'.";
        return false;
    }
    
    return true;
}

Settings::ImageSlot::CostFunction read_cost_function_name(const YAML::Node& node)
{
    std::string fn;
    try {
        fn = node.as<std::string>();
    }
    catch (YAML::TypedBadConversion<std::string>&) {
        throw UnrecognisedCostFunction(node.Mark());
    }

    if (fn == "none") {
        return Settings::ImageSlot::CostFunction_None;
    }
    else if (fn == "squared_distance" || fn == "ssd") {
        return Settings::ImageSlot::CostFunction_SSD;
    }
    else if (fn == "ncc") {
        return Settings::ImageSlot::CostFunction_NCC;
    }

    throw UnrecognisedCostFunction(node.Mark());
}

bool read_cost_function(const YAML::Node& node, Settings::ImageSlot::WeightedFunction& fn) {
    if (node["function"]) {
        try {
            fn.function = read_cost_function_name(node["function"]);
        }
        catch (UnrecognisedCostFunction& e) {
            LOG(Error) << e.what();
            return false;
        }
    }
    else {
        LOG(Error) << "Settings: line " << node.Mark().line << ", missing function";
        return false;
    }

    if (node["weight"]) {
        try {
            fn.weight = node["weight"].as<float>();
        }
        catch (YAML::TypedBadConversion<float>& e) {
            LOG(Error) << "Settings: line " << node.Mark().line << ", expected a weight";
            return false;
        }
    }
    else {
        fn.weight = 1.0f;
    }
    return true;
}

bool read_cost_functions(const YAML::Node& cost_functions, Settings::ImageSlot& slot) {
    if (!cost_functions) {
        return false;
    }

    // can be a list of cost functions
    if (cost_functions.IsSequence()) {
        slot.cost_functions.resize(cost_functions.size());
        for(size_t k = 0; k < cost_functions.size(); ++k) {
            if(!read_cost_function(cost_functions[k], slot.cost_functions[k])) {
                return false;
            }
        }
        return true;
    }

    // or it can be a scalar, i.e. the name of a function
    // (with default parameters) 
    try {
        slot.cost_functions[0].function = read_cost_function_name(cost_functions);
        slot.cost_functions[0].weight = 1.0f;
    }
    catch (UnrecognisedCostFunction& e) {
        LOG(Error) << e.what();
        return false;
    }
    return true;
}

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
    YAML::Node root;
    try {
        root = YAML::Load(str);
    }
    catch (YAML::ParserException& e) {
        LOG(Error) << "[YAML] " << e.what();
        return false;
    }
    
    if (root["pyramid_levels"] &&
        !read_value(root, "pyramid_levels", settings.num_pyramid_levels))
        return false;

    if (root["pyramid_stop_level"] &&
        !read_value(root, "pyramid_stop_level", settings.pyramid_stop_level))
        return false;

    if (root["step_size"] &&
        !read_value(root, "step_size", settings.step_size))
        return false;

    if (root["regularization_weight"] &&
        !read_value(root, "regularization_weight", settings.regularization_weight))
        return false;

    auto block_size = root["block_size"];
    if (block_size) {
        if (!block_size.IsSequence() || block_size.size() != 3) {
            LOG(Error) << "Settings: 'block_size', expected an array of 3 integers.";
            return false;
        }
        try {
            settings.block_size = {
                block_size[0].as<int>(),
                block_size[1].as<int>(),
                block_size[2].as<int>()
            };
        }
        catch (YAML::TypedBadConversion<int>&) {
            LOG(Error) << "Settings: 'block_size', expected an array of 3 integers.";
            return false;
        }
    }

    if (root["block_energy_epsilon"] &&
        !read_value(root, "block_energy_epsilon", settings.block_energy_epsilon))
        return false;

    if (root["constraints_weight"] &&
        !read_value(root, "constraints_weight", settings.constraints_weight))
        return false;

    auto image_slots = root["image_slots"];
    if (image_slots && image_slots.IsSequence()) {
        for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
            std::string is = std::to_string((long long int)i);

            auto slot = image_slots[is];
            if (slot.IsMap()) {
                if (slot["cost_function"] &&
                    !read_cost_functions(slot["cost_function"], settings.image_slots[i]))
                    return false;

                if (slot["resampler"] &&
                    !read_value(slot, "resampler", settings.image_slots[i].resample_method))
                    return false;

                if (slot["normalize"] &&
                    !read_value(slot, "normalize", settings.image_slots[i].normalize))
                    return false;
            }
        }
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

