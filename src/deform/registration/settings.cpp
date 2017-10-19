#include "settings.h"

#include <framework/json/json.h>

/*
    "pyramid":
    {
        "pyramid_levels": 6,
        "max_resolution": 0,
    },

    "constraints":
    {
        "weight": 1000
    },

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

    "optimizer":
    {
        "block_size": [12, 12, 12],
        "step_size": 0.5,
    }
    */


// Return true on success, false on failure
template<typename T>
bool read_value(const json::JsonObject& obj, const char* name, T& out);

template<>
bool read_value<int>(const json::JsonObject& obj, const char* name, int& out)
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
bool read_value<bool>(const json::JsonObject& obj, const char* name, bool& out)
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
bool read_value<bool>(const json::JsonObject& obj, const char* name, float& out)
{
    if (!obj[name].is_number())
    {
        LOG(Error, "Settings: '%s', expected boolean\n", name);
        return false;
    }
    out = obj[name].as_float();
    return true;
}

template<>
bool read_value<Settings::ImageSlot::CostFunction>(const json::JsonObject& obj, 
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
    else
    {
        LOG(Error, "Settings: Unrecognized value '%s'.\n", fn.c_str());
        return false;
    }
    
    return true;
}
template<>
bool read_value<Settings::ImageSlot::ResampleMethod>(const json::JsonObject& obj, 
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

bool load_registration_settings(const char* parameter_file, Settings& settings)
{
    // Defaults
    settings = Settings();

    json::JsonReader reader;
    json::JsonObject root;
    if (!reader.read_file(parameter_file, root))
    {
        LOG(Error, "Json: %s\n", reader.error_message().c_str());
        return false;
    }

    auto pyramid = root["pyramid"];
    if (pyramid.is_object())
    {
        if (!pyramid["pyramid_levels"].is_null() &&
            !read_value(pyramid, "pyramid_levels", settings.num_pyramid_levels))
            return false;

        if (!pyramid["start_level"].is_null() &&
            !read_value(pyramid, "start_level", settings.pyramid_start_level))
            return false;
    }

    auto image_slots = root["image_slots"];
    if (image_slots.is_object())
    {
        for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i)
        {
            std::string is = std::to_string(i);

            auto slot = image_slots[is];
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

    auto optimizer = root["optimizer"];
    if (optimizer.is_object())
    {
        if (!optimizer["step_size"].is_null() &&
            !read_value(optimizer, "step_size", settings.step_size))
            return false;

        auto block_size = optimizer["block_size"];
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
    }


    #ifdef DF_ENABLE_VOXEL_CONSTRAINTS
        auto constraints = root["constraints"];
        if (constraints.is_object())
        {
            if (read_variable(constraints, "weight", settings.constraint_weight))
                return false;
        }
    #endif


    return true;
}
