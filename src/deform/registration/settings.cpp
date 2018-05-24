#include "settings.h"

#include <nlohmann/json.hpp>
#include <stk/common/log.h>

#include <fstream>

using json = nlohmann::json;

/*
    "pyramid_levels": 6,
    "pyramid_stop_level": 0,

    "constraints_weight": 1000,

    "block_size": [12, 12, 12],
    "block_energy_epsilon": 0.001,
    "step_size": 0.5,
    "regularization_weight": 0.05,

    "image_slots":
    {
        "0": 
        { 
            "name": "water",
            "resampler": "gaussian", 
            "normalize": true, 
            "cost_function": "squared_distance"
        },
        "1": 
        {
            "name": "fat",
            "resampler": "gaussian", 
            "normalize": true,
            "cost_function": "squared_distance"
        },
        "2": 
        {
            "name": "sfcm",
            "resampler": "gaussian", 
            "normalize": true,
            "cost_function": "squared_distance"
        },
    },

    */


// Return true on success, false on failure
template<typename T>
bool read_value(const json& obj, const char* name, T& out);

template<>
bool read_value<int>(const json& obj, const char* name, int& out)
{
    if (!obj[name].is_number()) {
        LOG(Error) << "Settings: '" << name << "', expected integer";
        return false;
    }
    out = obj[name].get<int>();
    return true;
}
template<>
bool read_value<float>(const json& obj, const char* name, float& out)
{
    if (!obj[name].is_number()) {
        LOG(Error) << "Settings: '" << name << "', expected float";
        return false;
    }
    out = obj[name].get<float>();
    return true;
}
template<>
bool read_value<double>(const json& obj, const char* name, double& out)
{
    if (!obj[name].is_number()) {
        LOG(Error) << "Settings: '" << name << "', expected double";
        return false;
    }
    out = obj[name].get<double>();
    return true;
}
template<>
bool read_value<bool>(const json& obj, const char* name, bool& out)
{
    if (!obj[name].is_boolean()) {
        LOG(Error) << "Settings: '" << name << "', expected boolean";
        return false;
    }
    out = obj[name].get<bool>();
    return true;
}

template<>
bool read_value<Settings::ImageSlot::CostFunction>(const json& obj, 
    const char* name, Settings::ImageSlot::CostFunction& out)
{
    if (!obj[name].is_string()) {
        LOG(Error) << "Settings: '" << name << "', expected string";
        return false;
    }

    std::string fn = obj[name].get<std::string>();
    if (fn == "none") {
        out = Settings::ImageSlot::CostFunction_None;
    }
    else if (fn == "squared_distance") {
        out = Settings::ImageSlot::CostFunction_SSD;
    }
    else if (fn == "ncc") {
        out = Settings::ImageSlot::CostFunction_NCC;
    }
    else {
        LOG(Error) << "Settings: Unrecognized value '" << fn << "'.";
        return false;
    }
    
    return true;
}
template<>
bool read_value<Settings::ImageSlot::ResampleMethod>(const json& obj, 
    const char* name, Settings::ImageSlot::ResampleMethod& out)
{
    if (!obj[name].is_string()) {
        LOG(Error) << "Settings: '" << name << "', expected string";
        return false;
    }

    std::string fn = obj[name].get<std::string>();
    if (fn == "gaussian") {
        out = Settings::ImageSlot::Resample_Gaussian;
    }
    else {
        LOG(Error) << "Settings: Unrecognized value '" << fn << "'.";
        return false;
    }
    
    return true;
}

const char* cost_function_to_str(Settings::ImageSlot::CostFunction fn)
{
    switch (fn) {
    case Settings::ImageSlot::CostFunction_SSD:
        return "squared_distance";
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
        auto& slot = settings.image_slots[i];

        // Dont print unused slots
        if (slot.cost_function == Settings::ImageSlot::CostFunction_None)
            continue;

        LOG(Info) << "image_slot[" << i << "] = {";
        LOG(Info) << "  cost_function = " << cost_function_to_str(slot.cost_function);        
        LOG(Info) << "  resample_method = " << resample_method_to_str(slot.resample_method);        
        LOG(Info) << "  normalize = " << (slot.normalize ? "true" : "false");        
        LOG(Info) << "}";
    }
}

bool parse_registration_settings(const char* parameter_file, Settings& settings)
{
    // Defaults
    settings = Settings();

    std::ifstream f(parameter_file, std::ifstream::in);
    if (!f.is_open()) {
        LOG(Error) << "[Settings] Failed to open file '" << parameter_file << "'";
        return false;
    }


    json root;
    try {
        f >> root;
    }
    catch(json::parse_error err) {
        LOG(Error) << "[Json] " << err.what();
        return false;
    }
    
    if (!root["pyramid_levels"].is_null() &&
        !read_value(root, "pyramid_levels", settings.num_pyramid_levels))
        return false;

    if (!root["pyramid_stop_level"].is_null() &&
        !read_value(root, "pyramid_stop_level", settings.pyramid_stop_level))
        return false;

    if (!root["step_size"].is_null() &&
        !read_value(root, "step_size", settings.step_size))
        return false;

    if (!root["regularization_weight"].is_null() &&
        !read_value(root, "regularization_weight", settings.regularization_weight))
        return false;

    auto& block_size = root["block_size"];
    if (!block_size.is_null()) {
        if (!block_size.is_array() || 
            block_size.size() != 3 ||
            !block_size[0].is_number() ||
            !block_size[1].is_number() ||
            !block_size[2].is_number()) {
            LOG(Error) << "Settings: 'block_size', expected an array of 3 integers.";
            return false;
        }

        settings.block_size = {
            block_size[0].get<int>(),
            block_size[1].get<int>(),
            block_size[2].get<int>()
        };
    }

    if (!root["block_energy_epsilon"].is_null() &&
        !read_value(root, "block_energy_epsilon", settings.block_energy_epsilon))
        return false;

    if (!root["constraints_weight"].is_null() &&
        !read_value(root, "constraints_weight", settings.constraints_weight))
        return false;

    auto& image_slots = root["image_slots"];
    if (image_slots.is_object()) {
        for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
            std::string is = std::to_string((long long int)i);

            auto& slot = image_slots[is.c_str()];
            if (slot.is_object()) {
                if (!slot["cost_function"].is_null() &&
                    !read_value(slot, "cost_function", settings.image_slots[i].cost_function))
                    return false;

                if (!slot["resampler"].is_null() &&
                    !read_value(slot, "resampler", settings.image_slots[i].resample_method))
                    return false;

                if (!slot["normalize"].is_null() &&
                    !read_value(slot, "normalize", settings.image_slots[i].normalize))
                    return false;
            }
        }
    }

    print_registration_settings(settings);

    return true;
}
