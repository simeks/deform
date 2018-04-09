#include "settings.h"

#include <framework/debug/log.h>
#include <framework/json/json.h>
#include <framework/json/json_object.h>

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
bool read_value(const JsonObject& obj, const char* name, T& out);

template<>
bool read_value<int>(const JsonObject& obj, const char* name, int& out)
{
    if (!obj[name].is_number())
    {
        LOG(Error, "Settings: '%s', expected integer\n", name);
        return false;
    }
    out = obj[name].as_int();
    return true;
}
template<>
bool read_value<float>(const JsonObject& obj, const char* name, float& out)
{
    if (!obj[name].is_number())
    {
        LOG(Error, "Settings: '%s', expected float\n", name);
        return false;
    }
    out = obj[name].as_float();
    return true;
}
template<>
bool read_value<double>(const JsonObject& obj, const char* name, double& out)
{
    if (!obj[name].is_number())
    {
        LOG(Error, "Settings: '%s', expected double\n", name);
        return false;
    }
    out = obj[name].as_double();
    return true;
}
template<>
bool read_value<bool>(const JsonObject& obj, const char* name, bool& out)
{
    if (!obj[name].is_bool())
    {
        LOG(Error, "Settings: '%s', expected boolean\n", name);
        return false;
    }
    out = obj[name].as_bool();
    return true;
}

template<>
bool read_value<Settings::ImageSlot::CostFunction>(const JsonObject& obj, 
    const char* name, Settings::ImageSlot::CostFunction& out)
{
    if (!obj[name].is_string())
    {
        LOG(Error, "Settings: '%s', expected string\n", name);
        return false;
    }

    std::string fn = obj[name].as_string();
    if (fn == "none")
    {
        out = Settings::ImageSlot::CostFunction_None;
    }
    else if (fn == "squared_distance")
    {
        out = Settings::ImageSlot::CostFunction_SSD;
    }
    else if (fn == "ncc")
    {
        out = Settings::ImageSlot::CostFunction_NCC;
    }
    else
    {
        LOG(Error, "Settings: Unrecognized value '%s'.\n", fn.c_str());
        return false;
    }
    
    return true;
}
template<>
bool read_value<Settings::ImageSlot::ResampleMethod>(const JsonObject& obj, 
    const char* name, Settings::ImageSlot::ResampleMethod& out)
{
    if (!obj[name].is_string())
    {
        LOG(Error, "Settings: '%s', expected string\n", name);
        return false;
    }

    std::string fn = obj[name].as_string();
    if (fn == "gaussian")
    {
        out = Settings::ImageSlot::Resample_Gaussian;
    }
    else
    {
        LOG(Error, "Settings: Unrecognized value '%s'.\n", fn.c_str());
        return false;
    }
    
    return true;
}

const char* cost_function_to_str(Settings::ImageSlot::CostFunction fn)
{
    switch (fn)
    {
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
    switch (fn)
    {
        case Settings::ImageSlot::Resample_Gaussian:
            return "gaussian";
    };
    return "none";
}

void print_registration_settings(const Settings& settings)
{
    LOG(Info, "Settings:\n");
    LOG(Info, "pyramid_stop_level=%d\n",settings.pyramid_stop_level);
    LOG(Info, "num_pyramid_levels=%d\n",settings.num_pyramid_levels);
    LOG(Info, "block_size=[%d, %d, %d]\n", 
        settings.block_size.x, settings.block_size.y, settings.block_size.z);
    LOG(Info, "block_energy_epsilon=%f\n", settings.block_energy_epsilon);
    LOG(Info, "step_size=%f\n", settings.step_size);
    LOG(Info, "regularization_weight=%f\n", settings.regularization_weight);
    
    #ifdef DF_ENABLE_VOXEL_CONSTRAINTS
        LOG(Info, "constraints_weight=%f\n", settings.constraints_weight);
    #endif

    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i)
    {
        auto& slot = settings.image_slots[i];

        // Dont print unused slots
        if (slot.cost_function == Settings::ImageSlot::CostFunction_None)
            continue;

        LOG(Info, "image_slot[%d]={\n", i);
        LOG(Info, "\tcost_function=%s\n", cost_function_to_str(slot.cost_function));        
        LOG(Info, "\tresample_method=%s\n", resample_method_to_str(slot.resample_method));        
        LOG(Info, "\tnormalize=%s\n", slot.normalize ? "true" : "false");        
        LOG(Info, "}\n");
    }

}

bool parse_registration_settings(const char* parameter_file, Settings& settings)
{
    // Defaults
    settings = Settings();

    JsonReader reader;
    JsonObject root;
    if (!reader.read_file(parameter_file, root))
    {
        LOG(Error, "Json: %s\n", reader.error_message().c_str());
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
    if (!block_size.is_null())
    {
        if (!block_size.is_array() || 
            block_size.size() != 3 ||
            !block_size[0].is_number() ||
            !block_size[1].is_number() ||
            !block_size[2].is_number())
        {
            LOG(Error, "Settings: 'block_size', expected a array of 3 integers.\n");
            return false;
        }

        settings.block_size = {
            block_size[0].as_int(),
            block_size[1].as_int(),
            block_size[2].as_int()
        };
    }

    if (!root["block_energy_epsilon"].is_null() &&
        !read_value(root, "block_energy_epsilon", settings.block_energy_epsilon))
        return false;

    #ifdef DF_ENABLE_VOXEL_CONSTRAINTS
        if (!root["constraints_weight"].is_null() &&
            !read_value(root, "constraints_weight", settings.constraints_weight))
            return false;
    #endif

    auto& image_slots = root["image_slots"];
    if (image_slots.is_object())
    {
        for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i)
        {
            std::string is = std::to_string((long long int)i);

            auto& slot = image_slots[is.c_str()];
            if (slot.is_object())
            {
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
