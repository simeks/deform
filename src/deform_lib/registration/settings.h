#pragma once

#include "../config.h"

#include <stk/math/int3.h>

struct Settings
{
    // Settings for a specific image slot, each image pair (i.e. fixed and moving) is considered
    //  a slot, so say we want to register fat and water:
    //      fixed_water <- moving_water
    //      fixed_fat <- moving_fat
    //  then we have 2 slots, one for each modality 
    struct ImageSlot
    {
        enum CostFunction
        {
            CostFunction_None = 0,
            CostFunction_SSD,
            CostFunction_NCC
        };

        enum ResampleMethod
        {
            Resample_Gaussian // Applies a gaussian filter before downsampling
        };

        // Cost function to apply on this image pair
        CostFunction cost_function;

        // Specifies which resampler to use when building the pyramid
        ResampleMethod resample_method;

        // Indicates if images in this slot should be normalized
        bool normalize;

        ImageSlot() :
            cost_function(CostFunction_None),
            resample_method(Resample_Gaussian),
            normalize(true) {}
    };

    // Which level to start registration on
    // 0 being original resolution
    int pyramid_stop_level;
    // Size of pyramid
    int num_pyramid_levels;

    ImageSlot image_slots[DF_MAX_IMAGE_PAIR_COUNT];

    // Optimizer specific settings (Blocked graph cut):

    // Block size, (0,0,0) is the same as using only a single large block
    int3 block_size;
    // Epsilon used for termination
    double block_energy_epsilon;
    // Step size in [mm]
    float step_size;
    // Only considered if no weight map is given
    float regularization_weight;

    // Only applicable when constraints are present
    // High weight means harder constraints, a high value (>1000.0f) will act as hard constraints
    float constraints_weight;

    Settings() :
        pyramid_stop_level(0),
        num_pyramid_levels(6),
        block_size(int3{12, 12, 12}),
        block_energy_epsilon(0.01f),
        step_size(0.5f),
        regularization_weight(0.05f),
        constraints_weight(1000.0f)
    {}
};

// Prints the settings to the log in a human-readable format
void print_registration_settings(const Settings& settings);

// Return true if parsing was successful, false if not
bool parse_registration_settings(const std::string& parameter_file, Settings& settings);

