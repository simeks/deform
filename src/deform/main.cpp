#include "config.h"
#include "jacobian.h"
#include "registration/registration_engine.h"
#include "registration/settings.h"
#include "registration/transform.h"
#include "registration/volume_pyramid.h"

#include <framework/debug/assert.h>
#include <framework/debug/log.h>
#include <framework/filters/normalize.h>
#include <framework/filters/resample.h>
#include <framework/platform/file_path.h>
#include <framework/platform/timer.h>
#include <framework/profiler/microprofile.h>
#include <framework/volume/volume.h>
#include <framework/volume/volume_helper.h>
#include <framework/volume/stb.h>
#include <framework/volume/vtk.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <omp.h>
#include <string>
#include <string.h>
#include <vector>

#ifdef DF_ENABLE_BENCHMARK
    int run_benchmark(int argc, char* argv[]);
#endif

int run_regularize(int argc, char* argv[]);

namespace
{
    struct Args
    {
        const char* param_file;
        
        const char* fixed_files[DF_MAX_IMAGE_PAIR_COUNT];
        const char* moving_files[DF_MAX_IMAGE_PAIR_COUNT];

        const char* initial_deformation;

        #ifdef DF_ENABLE_VOXEL_CONSTRAINTS
            const char* constraint_mask;
            const char* constraint_values;
        #endif // DF_ENABLE_VOXEL_CONSTRAINTS
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
        #ifdef DF_ENABLE_VOXEL_CONSTRAINTS
                << "-constraint_mask <file> : Filename for constraint mask" << std::endl
                << "-constraint_values <file> : Filename for constraint values" << std::endl
        #endif // DF_ENABLE_VOXEL_CONSTRAINTS
                << "-p <file> : Filename of the parameter file (required)." << std::endl
                << "--help : Shows this help section." << std::endl
                << "*Requires a matching number of fixed and moving images";
        exit(1);
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
    #ifdef DF_ENABLE_VOXEL_CONSTRAINTS
                else if (key == "constraint_mask" || // Support both keys for backwards compatibility
                         key == "constraints_mask")
                {
                    if (++i >= argc) 
                        print_help_and_exit("Missing arguments");
                    args.constraint_mask = argv[i];
                }
                else if (key == "constraint_values" ||
                         key == "constraints_values")
                {
                    if (++i >= argc) 
                        print_help_and_exit("Missing arguments");
                    args.constraint_values = argv[i];
                }
    #endif // DF_ENABLE_VOXEL_CONSTRAINTS
                else
                {
                    std::string err = std::string("Unrecognized option: " + token);
                    print_help_and_exit(err.c_str());
                }
            }
            else
            {
                std::string err = std::string("Unrecognized option: " + token);
                print_help_and_exit(err.c_str());
            }
            ++i;
        }
    }
}

// Identifies and loads the given file
// file : Filename
// Returns the loaded volume, if load failed the returned volume will be flagged as invalid 
Volume load_volume(const std::string& file)
{
    FilePath path(file);
    std::string ext = path.extension();
    
    // To lower case
    #ifdef DF_PLATFORM_LINUX
        for (size_t i = 0; i < ext.size(); ++i) ext[i] = (char)::tolower(ext[i]);
    #else
        std::transform(ext.begin(), ext.end(), ext.begin(), [](char c) -> char { return (char)::tolower(c); });
    #endif

    if (ext == "vtk")
    {
        vtk::Reader reader;
        Volume vol = reader.execute(file.c_str());
        if (!vol.valid())        
        {
            LOG(Error, "Failed to read '%s': %s\n", file.c_str(), reader.last_error());
        }
        return vol;
    }
    else if (ext == "png")
    {
        std::string err;
        Volume vol = stb::read_image(file.c_str());
        if (!vol.valid())
        {
            LOG(Error, "Failed to read '%s': %s\n", file.c_str(), stb::last_read_error());
        }
        return vol;
    }
    else
    {
        LOG(Error, "Unsupported file extension: '%s'\n", file.c_str());
    }
    // Returning an "invalid" volume
    return Volume();
}

int run_transform(int argc, char* argv[])
{
    // Usage:
    // ./deform transform <src> <deformation> <out> [-i <nn/linear>]

    if (argc < 5)
    {
        std::cout << "Usage: " << argv[0] << " transform <src> <deformation> <out> [-i <nn/linear>]" << std::endl;
        return 1;
    }

    Volume src = load_volume(argv[2]);
    if (!src.valid())
        return 1;

    Volume def = load_volume(argv[3]);
    if (!def.valid())
        return 1;

    transform::Interp interp = transform::Interp_Linear;

    // TODO: Quick fix, include when refactoring command-line args
    if (argc == 7 && strcmp(argv[5], "-i") == 0 && strcmp(argv[6], "nn") == 0)
    {
        interp = transform::Interp_NN;
    }

    // TODO: Verify that def is float3

    Volume result = transform_volume(src, def, interp);
    vtk::write_volume(argv[4], result);
    
    return 0;
}

int run_jacobian(int argc, char* argv[])
{
    // Usage:
    // ./deform jacobian <source> <deformation> <out>

    if (argc < 5)
    {
        std::cout << "Usage: " << argv[0] << " jacobian <source> <deformation> <out>" << std::endl;
        return 1;
    }

    Volume src = load_volume(argv[2]);
    if (!src.valid())
        return 1;

    Volume def = load_volume(argv[3]);
    if (!def.valid())
        return 1;

    Volume jac = calculate_jacobian(src, def);
    vtk::write_volume(argv[4], jac);
    
    return 0;
}


int main(int argc, char* argv[])
{
    timer::initialize();

    #if MICROPROFILE_ENABLED
        MicroProfileOnThreadCreate("main");

        // Name all OpenMP threads for profiler
        auto main_thread = omp_get_thread_num();
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < 8; ++i)
        {
            if (omp_get_thread_num() != main_thread)
                MicroProfileOnThreadCreate("omp_worker");
        }

        MicroProfileSetEnableAllGroups(true);
        MicroProfileSetForceMetaCounters(true);
        //MicroProfileStartContextSwitchTrace();
    #endif // MICROPROFILE_ENABLED


    #ifdef DF_BUILD_DEBUG
        LOG(Warning, "Running debug build!\n");
    #endif

    if (argc >= 2 && strcmp(argv[1], "transform") == 0)
        return run_transform(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "regularize") == 0)
        return run_regularize(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "jacobian") == 0)
        return run_jacobian(argc, argv);
    
    #ifdef DF_ENABLE_BENCHMARK
        if (argc >= 2 && strcmp(argv[1], "benchmark") == 0)
            return run_benchmark(argc, argv);
    #endif

    Args input_args = {0};
    parse_command_line(input_args, argc, argv);

    if (input_args.param_file == 0)
        print_help_and_exit();

    Settings settings;
    if (!parse_registration_settings(input_args.param_file, settings))
        return 1;

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

        auto& slot = settings.image_slots[i];
    
        if (slot.normalize)
        {
            if (fixed.voxel_type() == voxel::Type_Float &&
                moving.voxel_type() == voxel::Type_Float)
            {
                fixed = filters::normalize<float>(fixed, 0.0f, 1.0f);
                moving = filters::normalize<float>(moving, 0.0f, 1.0f);
            }
            else if (fixed.voxel_type() == voxel::Type_Double &&
                     moving.voxel_type() == voxel::Type_Double)
            {
                fixed = filters::normalize<double>(fixed, 0.0, 1.0);
                moving = filters::normalize<double>(moving, 0.0, 1.0);
            }
            else
            {
                LOG(Error, "Normalize only supported on volumes of type float or double\n");
                return 1;
            }
        }
        
        // It's the only available fn for now
        Volume (*downsample_fn)(const Volume&, float) = filters::downsample_volume_gaussian;

        moving_volumes.push_back(moving);
        engine.set_image_pair(i, fixed, moving, downsample_fn);
    }

    if (input_args.initial_deformation)
    {
        Volume initial_deformation = load_volume(input_args.initial_deformation);
        if (!initial_deformation.valid()) return 1;

        engine.set_initial_deformation(initial_deformation);
    }

#ifdef DF_ENABLE_VOXEL_CONSTRAINTS
    if (input_args.constraint_mask && input_args.constraint_values)
    {
        Volume constraint_mask = load_volume(input_args.constraint_mask);
        if (!constraint_mask.valid()) return 1;

        Volume constraint_values = load_volume(input_args.constraint_values);
        if (!constraint_values.valid()) return 1;

        engine.set_voxel_constraints(constraint_mask, constraint_values);
    }
    else if (input_args.constraint_mask || input_args.constraint_values)
    {
        // Just a check to make sure the user didn't forget something
        LOG(Warning, "No constraints used, to use constraints, specify both a mask and a vectorfield\n");
    }
#endif // DF_ENABLE_VOXEL_CONSTRAINTS

    if (!engine.validate_input())
        exit(1);

    double t_start = timer::seconds();
    Volume def = engine.execute();
    double t_end = timer::seconds();
    int elapsed = int(round(t_end - t_start));
    LOG(Info, "Registration completed in %d:%02d\n", elapsed / 60, elapsed % 60);

    vtk::write_volume("result_def.vtk", def);

    Volume result = transform_volume(moving_volumes[0], def);
    vtk::write_volume("result.vtk", result);


    #if MICROPROFILE_ENABLED
        MicroProfileDumpFileImmediately("profiler_dump.html", "profiler_dump.csv", NULL);
        MicroProfileShutdown();
    #endif // MICROPROFILE_ENABLED

    return 0;
}
