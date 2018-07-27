#include <deform_lib/arg_parser.h>
#include <deform_lib/config.h>
#include <deform_lib/defer.h>
#include <deform_lib/filters/resample.h>
#include <deform_lib/jacobian.h>
#include <deform_lib/registration/landmarks.h>
#include <deform_lib/registration/registration.h>
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
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <optional>
#include <string>
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
    args.add_option("fixed{i}",     "-f{i}",        "Path to the i:th fixed image", true);
    args.add_option("moving{i}",    "-m{i}",        "Path to the i:th moving image", true);
    args.add_option("output",       "-o, --output", "Path to the initial deformation field");
    args.add_group("Optional");
    args.add_option("init_deform",  "-d0", "Path to the initial deformation field");
    args.add_group();
    args.add_option("fixed_points", "-fp, --fixed-points", "Path to the fixed landmark points");
    args.add_option("moving_points", "-mp, --moving-points", "Path to the moving landmark points");
    args.add_group();
    args.add_option("constraint_mask", "--constraint_mask", "Path to the constraint mask");
    args.add_option("constraint_values", "--constraint_values", "Path to the constraint values");
    args.add_group();
    args.add_flag("do_jacobian", "-j, --jacobian",  "Enable output of the resulting jacobian");
    args.add_flag("do_transform", "-t, --transform",  "Will output the transformed version of the first moving volume");
    args.add_group();
    args.add_option("num_threads", "--num-threads", "Maximum number of threads");

    if (!args.parse()) {
        return 1;
    }

    std::string param_file = args.get<std::string>("param_file", "");

    Settings settings; // Default settings
    if (!param_file.empty()) {
        LOG(Info) << "Running with parameter file: '" << param_file << "'";
        if (!parse_registration_file(param_file, settings))
            return 1;
    }
    else {
        LOG(Info) << "Running with default settings.";
    }

    std::vector<stk::Volume> fixed_volumes;
    std::vector<stk::Volume> moving_volumes;

    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
        std::string fixed_id = "fixed" + std::to_string(i);
        std::string moving_id = "moving" + std::to_string(i);

        std::string fixed_file = args.get<std::string>(fixed_id, "");
        std::string moving_file = args.get<std::string>(moving_id, "");

        if (fixed_file == "" || moving_file == "")
            continue;

        fixed_volumes.push_back(stk::read_volume(fixed_file));
        moving_volumes.push_back(stk::read_volume(moving_file));

        LOG(Info) << "Fixed image [" << i << "]: '" << fixed_file << "'";
        LOG(Info) << "Moving image [" << i << "]: '" << moving_file << "'";
    }

    std::string init_deform_file = args.get<std::string>("init_deform", "");
    LOG(Info) << "Initial deformation '" << init_deform_file << "'";

    std::optional<stk::Volume> initial_deformation;
    if (!init_deform_file.empty()) {
        initial_deformation = stk::read_volume(init_deform_file.c_str());
        LOG(Info) << "Initial deformation '" << init_deform_file << "'";
    }

    std::string constraint_mask_file = args.get<std::string>("constraint_mask", "");
    std::string constraint_values_file = args.get<std::string>("constraint_values", "");

    LOG(Info) << "Constraint mask: '" << constraint_mask_file << "'";
    LOG(Info) << "Constraint values: '" << constraint_values_file << "'";

    std::optional<stk::Volume> constraint_mask;
    std::optional<stk::Volume> constraint_values;
    if (!constraint_mask_file.empty() && !constraint_values_file.empty()) {
        constraint_mask = stk::read_volume(constraint_mask_file.c_str());
        constraint_values = stk::read_volume(constraint_values_file.c_str());
    }
    else if (!constraint_mask_file.empty() || !constraint_values_file.empty()) {
        // Just a check to make sure the user didn't forget something
        LOG(Error) << "No constraints used, to use constraints, specify both a mask and a vectorfield";
        return 1;
    }

    std::string fixed_landmarks_file = args.get<std::string>("fixed_points", "");
    std::string moving_landmarks_file = args.get<std::string>("moving_points", "");

    LOG(Info) << "Fixed landmarks: '" << fixed_landmarks_file << "'";
    LOG(Info) << "Moving landmarks: '" << moving_landmarks_file << "'";

    std::optional<std::vector<float3>> fixed_landmarks;
    std::optional<std::vector<float3>> moving_landmarks;
    try{
        if (!fixed_landmarks_file.empty()) {
            fixed_landmarks = parse_landmarks_file(fixed_landmarks_file.c_str());
        }
        if (!moving_landmarks_file.empty()) {
            moving_landmarks = parse_landmarks_file(moving_landmarks_file.c_str());
        }
    }
    catch (ValidationError& e) {
        LOG(Error) << e.what();
        return 1;
    }

    stk::Volume def;
    try {
        def = registration(settings,
                           fixed_volumes,
                           moving_volumes,
                           fixed_landmarks,
                           moving_landmarks,
                           initial_deformation,
                           constraint_mask,
                           constraint_values,
                           args.get<int>("num_threads", 0));
    }
    catch (ValidationError& e) {
        LOG(Error) << e.what();
        return 1;
    }

    std::string out_file = args.get<std::string>("output", "result_def.vtk");
    LOG(Info) << "Writing deformation field to '" << out_file << "'";
    stk::write_volume(out_file.c_str(), def);

    if (args.is_set("do_jacobian")) {
        LOG(Info) << "Writing jacobian to 'result_jac.vtk'";
        stk::Volume jac = calculate_jacobian(def);
        stk::write_volume("result_jac.vtk", jac);
    }

    if (args.is_set("do_transform")) {
        LOG(Info) << "Writing transformed image to 'result.vtk'";
        stk::Volume t = transform_volume(moving_volumes[0], def);
        stk::write_volume("result.vtk", t);
    }

    return 0;
}

void print_version()
{
    std::cout << "VERSION 0" << std::endl;
}

int main(int argc, char* argv[])
{
    stk::log_init();
    defer{stk::log_shutdown();};

    stk::log_add_file("deform_log.txt", stk::Info);

    #ifdef _DEBUG
        LOG(Warning) << "Running debug build!";
    #endif

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
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
