#pragma once

#include <map>
#include <vector>

#include "../config.h"

#include <stk/math/int3.h>

class ValidationError : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;
};

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
            CostFunction_NCC,
            CostFunction_MI,
            CostFunction_Gradient_SSD,
        };

        enum ResampleMethod
        {
            Resample_Gaussian // Applies a gaussian filter before downsampling
        };

        // A function associated with a weight
        struct WeightedFunction {
            float weight = 1.0;
            CostFunction function = CostFunction_None;
            std::map<std::string, std::string> parameters;
        };

        // Cost functions to apply on this image pair
        std::vector<WeightedFunction> cost_functions;

        // Specifies which resampler to use when building the pyramid
        ResampleMethod resample_method;

        // Indicates if images in this slot should be normalized
        bool normalize;

        ImageSlot() :
            cost_functions{{1.0, CostFunction_SSD, {}}},
            resample_method{Resample_Gaussian},
            normalize{true} {}
    };

    struct Level
    {
        // Optimizer specific settings (Blocked graph cut):

        // Block size, (0,0,0) is the same as using only a single large block
        int3 block_size;

        // Epsilon used for termination
        double block_energy_epsilon;

        // Maximum number of iterations, -1 indicates an infinite number of iterations
        int max_iteration_count;

        // Only considered if no weight map is given
        float regularization_weight;
        float regularization_scale;
        float regularization_exponent;

        // Step size in [mm]
        float3 step_size;

        // Only applicable when constraints are present
        // High weight means harder constraints, a high value (>1000.0f) will act as hard constraints
        float constraints_weight;

        // Only applicable to landmark registration
        float landmarks_weight;

        // Decay of landmarks force
        float landmarks_decay;

        Level() :
            block_size(int3{16, 16, 16}),
            block_energy_epsilon(1e-7f),
            max_iteration_count(-1),
            regularization_weight(0.25f),
            regularization_scale(1.0f),
            regularization_exponent(2.0f),
            step_size({0.5f, 0.5f, 0.5f}),
            constraints_weight(1000.0f),
            landmarks_weight(1.0f),
            landmarks_decay(2.0f)
        {
        }
    };

    // Which level to start registration on
    // 0 being original resolution
    int pyramid_stop_level;
    // Size of pyramid
    int num_pyramid_levels;

    std::vector<Level> levels;

    // Last level for which landmarks are used
    int landmarks_stop_level;

    std::vector<ImageSlot> image_slots;

    Settings() :
        pyramid_stop_level(0),
        num_pyramid_levels(6),
        landmarks_stop_level(0)
    {
        levels.resize(num_pyramid_levels);
    }
};

// Prints the settings to the log in a human-readable format
void print_registration_settings(const Settings& settings, std::ostream& s);

// Return true if parsing was successful, false if not
bool parse_registration_file(const std::string& parameter_file, Settings& settings);

// Return true if parsing was successful, false if not
bool parse_registration_settings(const std::string& str, Settings& settings);

