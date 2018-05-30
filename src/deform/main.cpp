#include <deform_lib/arg_parser.h>
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

int run_jacobian(int argc, char* argv[]);
int run_regularize(int argc, char* argv[]);
int run_transform(int argc, char* argv[]);

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
    ArgParser args(argc, argv);
    args.add_positional("command", "registration, transform, regularize, jacobian");
    
    args.add_group();
    args.add_option("param_file",   "-p",           "Path to the parameter file");
    args.add_option("fixed{i}",     "-f{i}",        "Path to the i:th fixed image");
    args.add_option("moving{i}",    "-m{i}",        "Path to the i:th moving image");
    args.add_option("output",       "-o, --output", "Path to the initial deformation field");
    args.add_group("Optional");
    args.add_option("init_deform",  "-d0", "Path to the initial deformation field");
    args.add_group();
    args.add_option("constraint_mask", "--constraint_mask", "Path to the constraint mask");
    args.add_option("constraint_values", "--constraint_values", "Path to the constraint values");
    args.add_group();
    args.add_flag("do_jacobian", "-j, --jacobian",  "Enable output of the resulting jacobian");
    args.add_group();
    args.add_option("num_threads", "--num-threads", "Maximum number of threads");

    if (!args.parse()) {
        return 1;
    }

    int num_threads = args.get<int>("num_threads", 0);
    if (num_threads > 0) {
        DLOG(Info) << "Number of threads: " << num_threads;
        omp_set_num_threads(num_threads);
    }

    std::string param_file = args.get<std::string>("param_file", "");

    Settings settings; // Default settings
    if (!param_file.empty()) {
        if (!parse_registration_settings(param_file.c_str(), settings))
            return 1;
    }

    RegistrationEngine engine(settings);

    std::vector<std::string> fixed_files;
    std::vector<std::string> moving_files;

    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {

        fixed_files.push_back(args.get<std::string>(fixed_i.c_str(), ""));
        moving_files.push_back(args.get<std::string>(moving_i.c_str(), ""));
    }

    engine.initialize(image_pair_count);

    Volume moving_ref; // Reference volume for computing the jacobian

    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
        std::string fixed_i = std::string("fixed") + std::to_string(i);
        std::string moving_i = std::string("moving") + std::to_string(i);

        stk::Volume fixed = stk::read_volume(input_args.fixed_files[i]);
        if (!fixed.valid()) return 1;
        stk::Volume moving = stk::read_volume(input_args.moving_files[i]);
        if (!moving.valid()) return 1;

        if (!moving_ref.valid())
            moving_ref = moving;

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

    std::string init_deform_file = args.get<std::string>("init_deform", "");
    if (!init_deform_file.empty()) {
        stk::Volume initial_deformation = stk::read_volume(init_deform_file.c_str());
        if (!initial_deformation.valid()) return 1;

        engine.set_initial_deformation(initial_deformation);
    }

    std::string constraint_mask_file = args.get<std::string>("constraint_mask", "");
    std::string constraint_values_file = args.get<std::string>("constraint_values", "");

    if (!constraint_mask_file.empty() && !constraint_values_file.empty()) {
        stk::Volume constraint_mask = stk::read_volume(constraint_mask_file.c_str());
        if (!constraint_mask.valid()) return 1;

        stk::Volume constraint_values = stk::read_volume(constraint_values_file.c_str());
        if (!constraint_values.valid()) return 1;

        engine.set_voxel_constraints(constraint_mask, constraint_values);
    }
    else if (!constraint_mask_file.empty() || !constraint_values_file.empty()) {
        // Just a check to make sure the user didn't forget something
        FATAL() << "No constraints used, to use constraints, specify both a mask and a vectorfield";
    }

    if (!engine.validate_input())
        exit(1);

    double t_start = timer::seconds();
    stk::Volume def = engine.execute();
    double t_end = timer::seconds();
    int elapsed = int(round(t_end - t_start));
    LOG(Info) << "Registration completed in " << elapsed / 60 << ":" << std::setw(2) << std::setfill('0') << elapsed % 60;

    std::string out_file = args.get<std::string>("output", "result_def.vtk");
    stk::write_volume(out_file.c_str(), def);

    if (args.is_set("do_jacobian")) {
        stk::Volume jac = calculate_jacobian(moving_ref, def);
        stk::write_volume("result_jac.vtk", jac);
    }

    return 0;
}

void print_version()
{
    std::cout << "VERSION 0" << std::endl;
}

int main(int argc, char* argv[])
{
    timer::initialize();

    #ifdef _DEBUG
        LOG(Warning) << "Running debug build!";
    #endif

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], '-v') == 0 || strcmp(argv[i], '--version')) {
            print_version();
            return 0;
        }
    }

    if (argc >= 2 && strcmp(argv[1], "registration") == 0)
        return run_registration(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "transform") == 0)
        return run_transform(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "regularize") == 0)
        return run_regularize(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "jacobian") == 0)
        return run_jacobian(argc, argv);

    print_command_help(argv[0]);
    
    return 1;
}
