#include "config.h"
#include "config_file.h"
#include "registration/registration_engine.h"
#include "registration/transform.h"
#include "registration/volume_pyramid.h"

#include <framework/debug/assert.h>
#include <framework/debug/log.h>
#include <framework/filters/normalize.h>
#include <framework/filters/resample.h>
#include <framework/platform/file_path.h>
#include <framework/platform/timer.h>
#include <framework/volume/volume.h>
#include <framework/volume/volume_helper.h>
#include <framework/volume/stb.h>
#include <framework/volume/vtk.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct Args
{
    const char* param_file;
    
    const char* fixed_files[DF_MAX_IMAGE_PAIR_COUNT];
    const char* moving_files[DF_MAX_IMAGE_PAIR_COUNT];

    const char* initial_deformation;

#ifdef DF_ENABLE_HARD_CONSTRAINTS
    const char* constraint_mask;
    const char* constraint_values;
#endif // DF_ENABLE_HARD_CONSTRAINTS
};


void print_help_and_exit(const char* err = 0)
{
    if (err)
        std::cout << "Error: " << err << std::endl;

    std::cout << "Arguments:" << std::endl
              << "-f<i> <file> : Filename of the i:th fixed image (i < " 
                << DF_MAX_IMAGE_PAIR_COUNT << ")*." << std::endl
              << "-m<i> <file> : Filename of the i:th moving image (i < " 
                << DF_MAX_IMAGE_PAIR_COUNT << ")*." << std::endl
              << "-d0 <file> : Filename for initial deformation field" << std::endl
#ifdef DF_ENABLE_HARD_CONSTRAINTS
              << "-constraint_mask <file> : Filename for constraint mask" << std::endl
              << "-constraint_values <file> : Filename for constraint values" << std::endl
#endif // DF_ENABLE_HARD_CONSTRAINTS
              << "-p <file> : Filename of the parameter file (required)." << std::endl
              << "--help : Shows this help section." << std::endl
              << "*Requires a matching number of fixed and moving images";
    exit(0);
}
void parse_command_line(Args& args, int argc, char** argv)
{
    args = {0};

    /// Skip i=0 (name of executable)
    int i = 1;
    while (i < argc)
    {
        std::string token = argv[i];
        if (token[0] == '-')
        {
            int b = token[1] == '-' ? 2 : 1;
            std::string key = token.substr(b);

            if (key == "help")
            {
                print_help_and_exit();
            }
            else if (key == "p")
            {
                if (++i >= argc) 
                    print_help_and_exit("Missing arguments");
                args.param_file = argv[i];
            }
            else if (key[0] == 'f')
            {
                int img_index = std::stoi(key.substr(1));
                if (img_index >= DF_MAX_IMAGE_PAIR_COUNT)
                    print_help_and_exit();

                if (++i >= argc)
                    print_help_and_exit("Missing arguments");
                
                args.fixed_files[img_index] = argv[i];
            }
            else if (key[0] == 'm')
            {
                int img_index = std::stoi(key.substr(1));
                if (img_index >= DF_MAX_IMAGE_PAIR_COUNT)
                    print_help_and_exit();

                if (++i >= argc)
                    print_help_and_exit("Missing arguments");
                
                args.moving_files[img_index] = argv[i];
            }
            else if (key == "d0")
            {
                if (++i >= argc) 
                    print_help_and_exit("Missing arguments");
                args.initial_deformation = argv[i];
            }
#ifdef DF_ENABLE_HARD_CONSTRAINTS
            else if (key == "constraint_mask")
            {
                if (++i >= argc) 
                    print_help_and_exit("Missing arguments");
                args.constraint_mask = argv[i];
            }
            else if (key == "constraint_values")
            {
                if (++i >= argc) 
                    print_help_and_exit("Missing arguments");
                args.constraint_values = argv[i];
            }
#endif // DF_ENABLE_HARD_CONSTRAINTS
            else
            {
                print_help_and_exit("Unrecognized option");
            }
        }
        else
        {
            print_help_and_exit("Unrecognized option");
        }
        ++i;
    }
}
/// Returns true if parsing was successful, false if not
void parse_parameter_file(RegistrationEngine::Settings& settings, const char* file)
{
    // Assumes settings is filled with the default values beforehand

    ConfigFile cfg(file);

    if (cfg.keyExists("REGISTRATION_METHOD"))
    {
        LOG(Warning, "Parameter REGISTRATION_METHOD not applicable, ignoring.\n");
    }

    if (cfg.keyExists("NORMALIZE_IMAGES"))
    {
        LOG(Warning, "Parameter NORMALIZE_IMAGES not applicable (depends on image type), ignoring.\n");
    }

    settings.pyramid_levels = cfg.getValueOfKey<int>("PYRAMID_LEVELS", settings.pyramid_levels);
    settings.max_pyramid_level = cfg.getValueOfKey<int>("MAX_RESOLUTION", settings.max_pyramid_level);
    settings.step_size = cfg.getValueOfKey<float>("STEPSIZE", settings.step_size);
    settings.regularization_weight = cfg.getValueOfKey<float>("REGULARIZATION_WEIGHT", settings.regularization_weight);
    
    LOG(Info, "Settings:\n");
    LOG(Info, "pyramid_levels = %d\n", settings.pyramid_levels);
    LOG(Info, "max_pyramid_level = %d\n", settings.max_pyramid_level);
    LOG(Info, "step_size = %f\n", settings.step_size);
    LOG(Info, "regularization_weight = %f\n", settings.regularization_weight);
}

// Identifies and loads the given file
// file : Filename
// Returns the loaded volume, if load failed the returned volume will be flagged as invalid 
Volume load_volume(const std::string& file)
{
    FilePath path(file);
    std::string ext = path.extension();
    
    // To lower case
    std::transform(ext.begin(), ext.end(), ext.begin(), [](char c){ return (char)::tolower(c); });

    if (ext == "vtk")
    {
        vtk::Reader reader;
        Volume vol = reader.execute(file.c_str());
        if (!vol.valid())        
        {
            LOG(Error, "Failed to read image: %s\n", reader.last_error());
        }
        return vol;
    }
    else if (ext == "png")
    {
        std::string err;
        Volume vol = stb::read_image(file.c_str());
        if (!vol.valid())
        {
            LOG(Error, "Failed to read image: %s\n", stb::last_read_error());
        }
        return vol;
    }
    else
    {
        LOG(Error, "Unsupported file extension: '%s'\n", ext.c_str());
    }
    // Returning an "invalid" volume
    return Volume();
}

int main(int argc, char* argv[])
{
    timer::initialize();

    #ifdef DF_BUILD_DEBUG
        LOG(Warning, "Running debug build!\n");
    #endif

    Args input_args = {0};
    parse_command_line(input_args, argc, argv);

    if (input_args.param_file == 0)
        print_help_and_exit();

    RegistrationEngine::Settings settings;
    parse_parameter_file(settings, input_args.param_file);

    int image_pair_count = 0;
    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i)
    {
        if (input_args.fixed_files[i] && 
            input_args.moving_files[i] &&
            image_pair_count == i)
            ++image_pair_count;
    }

    if (image_pair_count == 0)
    {
        LOG(Error, "No (or invalid) input images, are you sure you (1) gave a matching \
            number of fixed and moving images, and (2) filled the slots incrementally (0, 1, ... n)?\n");
        return 1;
    }

    RegistrationEngine engine(settings);
    engine.initialize(image_pair_count);
    
    std::vector<Volume> moving_volumes;
    for (int i = 0; i < image_pair_count; ++i)
    {
        Volume fixed = load_volume(input_args.fixed_files[i]);
        if (!fixed.valid()) return 1;
        Volume moving = load_volume(input_args.moving_files[i]);
        if (!moving.valid()) return 1;

        // TODO: This should probably not be performed for all image types
        fixed = filters::normalize<float>(fixed, 0.0f, 1.0f);
        moving = filters::normalize<float>(moving, 0.0f, 1.0f);

        moving_volumes.push_back(moving);
        engine.set_image_pair(i, fixed, moving, filters::downsample_volume_gaussian);
    }

    if (input_args.initial_deformation)
    {
        Volume initial_deformation = load_volume(input_args.initial_deformation);
        if (!initial_deformation.valid()) return 1;

        engine.set_initial_deformation(initial_deformation);
    }

#ifdef DF_ENABLE_HARD_CONSTRAINTS
    if (input_args.constraint_mask && input_args.constraint_values)
    {
        Volume constraint_mask = load_volume(input_args.constraint_mask);
        if (!constraint_mask.valid()) return 1;

        Volume constraint_values = load_volume(input_args.constraint_values);
        if (!constraint_values.valid()) return 1;

        engine.set_hard_constraints(constraint_mask, constraint_values);
    }
    else if (input_args.constraint_mask || input_args.constraint_values)
    {
        // Just a check to make sure the user didn't forget something
        LOG(Warning, "No constraints used, to use constraints, specify both a mask and a vectorfield\n");
    }
#endif // DF_ENABLE_HARD_CONSTRAINTS

    if (!engine.validate_input())
        exit(1);

    Volume def = engine.execute();

    vtk::write_volume("result_def.vtk", def);

    Volume result = transform_volume(moving_volumes[0], def);
    vtk::write_volume("result.vtk", result);

    return 0;
}
