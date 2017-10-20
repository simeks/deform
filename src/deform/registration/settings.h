#pragma once

#include "config.h"

#include <framework/math/int3.h>

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
            CostFunction_SSD
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
    int pyramid_start_level;
    // Size of pyramid
    int num_pyramid_levels;

    ImageSlot image_slots[DF_MAX_IMAGE_PAIR_COUNT];

    // Optimizer specific settings (Blocked graph cut):

    // Block size, (0,0,0) is the same as using only a single large block
    int3 block_size;
    // Step size in [mm]
    float step_size;
    
#ifdef DF_ENABLE_VOXEL_CONSTRAINTS
    // Only applicable when constraints are present

    // High weight means harder constraints, a high value (>1000.0f) will act as hard constraints
    float constraints_weight;
#endif

    Settings() :
        pyramid_start_level(0),
        num_pyramid_levels(6),
        block_size(int3{12, 12, 12}),
        step_size(0.5f) 
#ifdef DF_ENABLE_VOXEL_CONSTRAINTS
        , constraint_weight(1000.0f)
#endif
    {}
};

// Return true if parsing was successful, false if not
bool parse_registration_settings(const char* parameter_file, Settings& settings);

