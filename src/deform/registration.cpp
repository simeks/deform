#include <stk/io/io.h>

#include <deform_lib/jacobian.h>

#include <deform_lib/registration/landmarks.h>
#include <deform_lib/registration/registration.h>
#include <deform_lib/registration/settings.h>
#include <deform_lib/registration/transform.h>

#include "deform/command.h"

#include <fstream>

bool RegistrationCommand::_parse_arguments(void)
{
    _args.add_positional("command", "registration, transform, regularize, jacobian");

    _args.add_group();
    _args.add_option("param_file",   "-p",           "Path to the parameter file");
    _args.add_option("fixed{i}",     "-f{i}",        "Path to the i:th fixed image", true);
    _args.add_option("moving{i}",    "-m{i}",        "Path to the i:th moving image", true);
    _args.add_option("output",       "-o, --output", "Path to the resulting displacement field");
    _args.add_group("Optional");
    _args.add_option("fixed_mask",   "-fm, --fixed-mask",   "Path to the fixed image mask");
    _args.add_option("moving_mask",  "-mm, --moving-mask", "Path to the moving image mask");
    _args.add_group();
    _args.add_option("init_deform",  "-d0", "Path to the initial displacement field");
    _args.add_group();
    _args.add_option("fixed_points", "-fp, --fixed-points", "Path to the fixed landmark points");
    _args.add_option("moving_points", "-mp, --moving-points", "Path to the moving landmark points");
    _args.add_group();
    _args.add_option("constraint_mask", "--constraint_mask", "Path to the constraint mask");
    _args.add_option("constraint_values", "--constraint_values", "Path to the constraint values");
    _args.add_group();
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    _args.add_option("regularization_map", "-rm, --regularization_map", "Path to a map of voxel-wise regularization terms");
    _args.add_group();
#endif
    _args.add_option("jacobian", "-j, --jacobian",  "Path to the resulting jacobian");
    _args.add_option("transform", "-t, --transform",  "Path to the transformed version of the first moving volume");
    _args.add_group();
    _args.add_option("num_threads", "--num-threads", "Maximum number of threads");

#ifdef DF_USE_CUDA
    _args.add_group();
    _args.add_flag("use_gpu", "--gpu", "Enables GPU supported registration");
#endif // DF_USE_CUDA

    return _args.parse();
}

int RegistrationCommand::_execute(void)
{
    std::string param_file = _args.get<std::string>("param_file", "");

    Settings settings; // Default settings

    if (!param_file.empty()) {
        std::ifstream f(param_file, std::ifstream::in);
        if (!f.is_open()) {
            LOG(Error) << "Failed to open parameter file '" << param_file << "'";
            return EXIT_FAILURE;
        }

        std::stringstream param_str;
        param_str << f.rdbuf();

        LOG(Info) << "Running with parameter file: '" << param_file << "'";
        try {
            parse_registration_settings(param_str.str(), settings);
        }
        catch (std::exception& e) {
            LOG(Error) << e.what();
            return EXIT_FAILURE;
        }

        // Print only contents of parameter file to Info
        LOG(Info) << "Parameters:" << std::endl << param_str.str();

        // Print all settings to Verbose
        
        std::stringstream settings_str;
        print_registration_settings(settings, settings_str);

        LOG(Verbose) << settings_str.rdbuf();
    }
    else {
        LOG(Info) << "Running with default settings.";
    }

    // Volumes
    std::vector<stk::Volume> fixed_volumes;
    std::vector<stk::Volume> moving_volumes;

    const int image_pair_count = _args.count_instances("fixed{i}");
    if (_args.count_instances("moving{i}") != image_pair_count) {
        LOG(Fatal) << "Mismatching number of fixed and moving images.";
        return EXIT_FAILURE;
    }

    if ((int) settings.image_slots.size() != image_pair_count) {
        LOG(Warning) << "Different number of images between input and settings!";
        settings.image_slots.resize(image_pair_count);
    }

    for (int i = 0; i < image_pair_count; ++i) {
        std::string fixed_id = "fixed" + std::to_string(i);
        std::string moving_id = "moving" + std::to_string(i);

        std::string fixed_file = _args.get<std::string>(fixed_id, "");
        std::string moving_file = _args.get<std::string>(moving_id, "");

        if (fixed_file == "" || moving_file == "")
            continue;

        fixed_volumes.push_back(stk::read_volume(fixed_file));
        moving_volumes.push_back(stk::read_volume(moving_file));

        LOG(Info) << "Fixed image [" << i << "]: '" << fixed_file << "'";
        LOG(Info) << "Moving image [" << i << "]: '" << moving_file << "'";
    }

    // Output
    std::string out_file = _args.get<std::string>("output", "result_def.vtk");
    LOG(Info) << "Output displacement file: '" << out_file << "'";

    // Masks
    const std::string fixed_mask_file = _args.get<std::string>("fixed_mask", "");
    const std::string moving_mask_file = _args.get<std::string>("moving_mask", "");

    stk::Volume fixed_mask;
    stk::Volume moving_mask;

    if (!fixed_mask_file.empty()) {
        fixed_mask = stk::read_volume(fixed_mask_file);
        if (!fixed_mask.valid()) 
            return EXIT_FAILURE;
    }

    if (!moving_mask_file.empty()) {
        moving_mask = stk::read_volume(moving_mask_file);
        if (!moving_mask.valid()) 
            return EXIT_FAILURE;
    }

    LOG(Info) << "Fixed mask: '" << fixed_mask_file << "'";
    LOG(Info) << "Moving mask: '" << moving_mask_file << "'";

    // Initial displacement
    std::string init_deform_file = _args.get<std::string>("init_deform", "");
    LOG(Info) << "Initial displacement: '" << init_deform_file << "'";

    stk::Volume initial_displacement;
    if (!init_deform_file.empty()) {
        initial_displacement = stk::read_volume(init_deform_file.c_str());
        if (!initial_displacement.valid()) 
            return EXIT_FAILURE;
    }

    // Constraints
    std::string constraint_mask_file = _args.get<std::string>("constraint_mask", "");
    std::string constraint_values_file = _args.get<std::string>("constraint_values", "");

    LOG(Info) << "Constraint mask: '" << constraint_mask_file << "'";
    LOG(Info) << "Constraint values: '" << constraint_values_file << "'";

    stk::Volume constraint_mask;
    stk::Volume constraint_values;
    if (!constraint_mask_file.empty() && !constraint_values_file.empty()) {
        constraint_mask = stk::read_volume(constraint_mask_file.c_str());
        if (!constraint_mask.valid()) 
            return EXIT_FAILURE;

        constraint_values = stk::read_volume(constraint_values_file.c_str());
        if (!constraint_values.valid()) 
            return EXIT_FAILURE;
    }
    else if (!constraint_mask_file.empty() || !constraint_values_file.empty()) {
        // Just a check to make sure the user didn't forget something
        LOG(Error) << "No constraints used, to use constraints, specify both a mask and a vector field";
        return EXIT_FAILURE;
    }

    // Landmarks
    std::string fixed_landmarks_file = _args.get<std::string>("fixed_points", "");
    std::string moving_landmarks_file = _args.get<std::string>("moving_points", "");

    LOG(Info) << "Fixed landmarks: '" << fixed_landmarks_file << "'";
    LOG(Info) << "Moving landmarks: '" << moving_landmarks_file << "'";

    std::vector<float3> fixed_landmarks;
    std::vector<float3> moving_landmarks;
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
        return EXIT_FAILURE;
    }

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    std::string regularization_map_file = _args.get<std::string>("regularization_map", "");
    LOG(Info) << "Regularization map: '" << regularization_map_file << "'";

    stk::Volume regularization_map;
    if (!regularization_map_file.empty()) {
        regularization_map = stk::read_volume(regularization_map_file.c_str());
        if (!regularization_map.valid()) 
            return EXIT_FAILURE;
    }
#endif

#ifdef DF_USE_CUDA
    bool use_gpu = _args.is_set("use_gpu");
#endif

    stk::Volume def;
    try {
        def = registration(settings,
                           fixed_volumes,
                           moving_volumes,
                           fixed_mask,
                           moving_mask,
                           fixed_landmarks,
                           moving_landmarks,
                           initial_displacement,
                           constraint_mask,
                           constraint_values,
                        #ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
                           regularization_map,
                        #endif
                           _args.get<int>("num_threads", 0)
                        #ifdef DF_USE_CUDA
                           , use_gpu
                        #endif
                           );
    }
    catch (std::exception& e) {
        LOG(Error) << e.what();
        return EXIT_FAILURE;
    }

    LOG(Info) << "Writing displacement field to '" << out_file << "'";
    stk::write_volume(out_file.c_str(), def);

    if (_args.is_set("jacobian")) {
        std::string jac_file = _args.get<std::string>("jacobian", "result_jac.vtk");
        LOG(Info) << "Writing jacobian to '" << jac_file << "'";
        stk::Volume jac = calculate_jacobian(def);
        stk::write_volume(jac_file, jac);
    }

    if (_args.is_set("transform")) {
        std::string transform_file = _args.get<std::string>("transform", "result.vtk");
        LOG(Info) << "Writing transformed image to '" << transform_file << "'";
        stk::Volume t = transform_volume(moving_volumes[0], def);
        stk::write_volume(transform_file, t);
    }

    return EXIT_SUCCESS;
}

