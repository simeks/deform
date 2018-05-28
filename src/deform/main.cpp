#include <deform_lib/config.h>
#include <deform_lib/filters/resample.h>
#include <deform_lib/jacobian.h>
#include <deform_lib/platform/file_path.h>
#include <deform_lib/platform/timer.h>
#include <deform_lib/registration/registration_engine.h>
#include <deform_lib/registration/settings.h>
#include <deform_lib/registration/transform.h>
#include <deform_lib/registration/volume_pyramid.h>

#include <stk/common/assert.h>
#include <stk/common/log.h>
#include <stk/filters/normalize.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>

#include <algorithm>
#include <iomanip>
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

        int num_threads;

        const char* constraint_mask;
        const char* constraint_values;
    };

    void print_help_and_exit(const char* exec, const char* err = 0)
    {
        if (err)
            std::cout << "Error: " << err << std::endl;

        std::cout << "Usage: " << exec << " registration {OPTIONS}" << std::endl << std::endl;

        std::cout << "COMMAND: registration" << std::endl << std::endl;
        std::cout << "OPTIONS:" << std::endl << std::endl;
        print_option_help("-p", "file", "Filename of the parameter file (required).");
        print_option_help("-f<i>", "file", "Filename of the i:th fixed image");
        print_option_help("-m<i>", "file", "Filename of the i:th moving image");

        std::cout << std::endl;
        std::cout << "OPTIONAL:" << std::endl << std::endl;
        print_option_help("-o, --output", "file", "Filename for the output deformation field");
        print_option_help("--constraint_mask", "file", "Filename for constraint mask");
        print_option_help("--constraint_values", "file", "Filename for constraint values");
        print_option_help("-d0", "file", "Filename for initial deformation field");
        print_flag_help("-j, --jacobian", "Enables output of the resulting jacobian");

        print_option_help("--num-threads", "num", "Maximum number of threads");

        print_flag_help("--help", "Displays this help text");

        std::cout << std::endl;
        std::cout << "Other commands: transform, regularize, jacobian" << std::endl;

        exit(1);
    }
    void parse_command_line(Args& args, int argc, char** argv)
    {
        args = {0};

        /// Skip i=0 (name of executable)
        int i = 1;
        while (i < argc) {
            std::string token = argv[i];
            if (token[0] == '-') {
                int b = token[1] == '-' ? 2 : 1;
                std::string key = token.substr(b);

                if (key == "help") {
                    print_help_and_exit(argv[0]);
                }
                else if (key == "p") {
                    if (++i >= argc) 
                        print_help_and_exit(argv[0], "Missing arguments");
                    args.param_file = argv[i];
                }
                else if (key[0] == 'f') {
                    int img_index = std::stoi(key.substr(1));
                    if (img_index >= DF_MAX_IMAGE_PAIR_COUNT)
                        print_help_and_exit(argv[0]);

                    if (++i >= argc)
                        print_help_and_exit(argv[0], "Missing arguments");
                    
                    args.fixed_files[img_index] = argv[i];
                }
                else if (key[0] == 'm') {
                    int img_index = std::stoi(key.substr(1));
                    if (img_index >= DF_MAX_IMAGE_PAIR_COUNT)
                        print_help_and_exit(argv[0]);

                    if (++i >= argc)
                        print_help_and_exit(argv[0], "Missing arguments");
                    
                    args.moving_files[img_index] = argv[i];
                }
                else if (key == "d0") {
                    if (++i >= argc) 
                        print_help_and_exit(argv[0], "Missing arguments");
                    args.initial_deformation = argv[i];
                }
                else if (key == "constraint_mask" || // Support both keys for backwards compatibility
                         key == "constraints_mask") {
                    if (++i >= argc) 
                        print_help_and_exit(argv[0], "Missing arguments");
                    args.constraint_mask = argv[i];
                }
                else if (key == "constraint_values" ||
                         key == "constraints_values") {
                    if (++i >= argc) 
                        print_help_and_exit(argv[0], "Missing arguments");
                    args.constraint_values = argv[i];
                }
                else if (key == "num-threads") {
                    if (++i >= argc)
                        print_help_and_exit(argv[0], "Missing arguments");
                    args.num_threads = std::stoi(argv[i]);
                }
                else {
                    std::string err = std::string("Unrecognized option: " + token);
                    print_help_and_exit(argv[0], err.c_str());
                }
            }
            else {
                std::string err = std::string("Unrecognized option: " + token);
                print_help_and_exit(argv[0], err.c_str());
            }
            ++i;
        }
    }
    void print_command_help(const char* exec)
    {
        std::cout << "Usage: " << exec << " COMMAND ..." << std::endl << std::endl;
        std::cout << "COMMANDS:" << std::endl << std::endl;

        std::cout << std::string(4, ' ') << std::setw(30) << std::left << "registration" 
                  << "Performs image registration" << std::endl;
        std::cout << std::string(4, ' ') << std::setw(30) << std::left << "transform" 
                  << "Transforms a volume with a given deformation field" << std::endl;
        std::cout << std::string(4, ' ') << std::setw(30) << std::left << "regularize" 
                  << "Regularizes a deformation field" << std::endl;
        std::cout << std::string(4, ' ') << std::setw(30) << std::left << "jacobian" 
                  << "Computes the jacobian determinants of a deformation field" << std::endl;
    }
}

int run_registration(int argc, char* argv[])
{
    
}

int main(int argc, char* argv[])
{
    if (argc >= 2 && strcmp(argv[1], "registration") == 0)
        return run_registration(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "transform") == 0)
        return run_transform(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "regularize") == 0)
        return run_regularize(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "jacobian") == 0)
        return run_jacobian(argc, argv);

    print_command_help(argv[0]);
}

int main2(int argc, char* argv[])
{
    timer::initialize();

    #ifdef DF_BUILD_DEBUG
        LOG(Warning) << "Running debug build!";
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

    if (input_args.num_threads > 0) {
        DLOG(Info) << "Number of threads: " << input_args.num_threads;
        omp_set_num_threads(input_args.num_threads);
    }

    if (input_args.param_file == 0)
        print_help_and_exit(argv[0]);

    Settings settings;
    if (!parse_registration_settings(input_args.param_file, settings))
        return 1;

    int image_pair_count = 0;
    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
        if (input_args.fixed_files[i] && 
            input_args.moving_files[i] &&
            image_pair_count == i)
            ++image_pair_count;
    }

    if (image_pair_count == 0) {
        LOG(Error) << "No (or invalid) input images, are you sure you (1) gave \
                       a matching number of fixed and moving images, and (2) \
                       filled the slots incrementally (0, 1, ... n)?";
        return 1;
    }

    RegistrationEngine engine(settings);
    engine.initialize(image_pair_count);
    
    std::vector<stk::Volume> moving_volumes;
    for (int i = 0; i < image_pair_count; ++i) {
        stk::Volume fixed = stk::read_volume(input_args.fixed_files[i]);
        if (!fixed.valid()) return 1;
        stk::Volume moving = stk::read_volume(input_args.moving_files[i]);
        if (!moving.valid()) return 1;

        auto& slot = settings.image_slots[i];
    
        if (slot.normalize) {
            if (fixed.voxel_type() == stk::Type_Float &&
                moving.voxel_type() == stk::Type_Float) {
                fixed = stk::normalize<float>(fixed, 0.0f, 1.0f);
                moving = stk::normalize<float>(moving, 0.0f, 1.0f);
            }
            else if (fixed.voxel_type() == stk::Type_Double &&
                     moving.voxel_type() == stk::Type_Double) {
                fixed = stk::normalize<double>(fixed, 0.0, 1.0);
                moving = stk::normalize<double>(moving, 0.0, 1.0);
            }
            else {
                LOG(Error) << "Normalize only supported on volumes of type float or double";
                return 1;
            }
        }
        
        // It's the only available fn for now
        auto downsample_fn = filters::downsample_volume_gaussian;

        moving_volumes.push_back(moving);
        engine.set_image_pair(i, fixed, moving, downsample_fn);
    }

    if (input_args.initial_deformation) {
        stk::Volume initial_deformation = stk::read_volume(input_args.initial_deformation);
        if (!initial_deformation.valid()) return 1;

        engine.set_initial_deformation(initial_deformation);
    }

    if (input_args.constraint_mask && input_args.constraint_values) {
        stk::Volume constraint_mask = stk::read_volume(input_args.constraint_mask);
        if (!constraint_mask.valid()) return 1;

        stk::Volume constraint_values = stk::read_volume(input_args.constraint_values);
        if (!constraint_values.valid()) return 1;

        engine.set_voxel_constraints(constraint_mask, constraint_values);
    }
    else if (input_args.constraint_mask || input_args.constraint_values) {
        // Just a check to make sure the user didn't forget something
        LOG(Warning) << "No constraints used, to use constraints, specify both a mask and a vectorfield";
    }

    if (!engine.validate_input())
        exit(1);

    double t_start = timer::seconds();
    stk::Volume def = engine.execute();
    double t_end = timer::seconds();
    int elapsed = int(round(t_end - t_start));
    LOG(Info) << "Registration completed in " << elapsed / 60 << ":" << std::setw(2) << std::setfill('0') << elapsed % 60;

    stk::write_volume("result_def.vtk", def);

    stk::Volume result = transform_volume(moving_volumes[0], def);
    stk::write_volume("result.vtk", result);

    stk::Volume jac = calculate_jacobian(moving_volumes[0], def);
    stk::write_volume("result_jac.vtk", jac);

    return 0;
}
