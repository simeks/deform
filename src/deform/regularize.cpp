#include <stk/io/io.h>
#include <stk/filters/gaussian_filter.h>

#include <deform_lib/regularize.h>
#include <deform_lib/filters/resample.h>

#include "deform/command.h"

#include <chrono>
#include <iomanip>

bool RegularisationCommand::_parse_arguments(void)
{
    _args.add_positional("command", "registration, transform, regularize, jacobian");
    _args.add_positional("deformation", "Path to the deformation field to regularize");
    _args.add_group();
    _args.add_option("output", "-o, --output", "Path to output (default: result_def.vtk)");
    _args.add_option("precision", "-p, --precision", "Precision (default: 0.5)");
    _args.add_option("pyramid_levels", "-l, --levels", "Number of pyramid levels (default: 6)");
    _args.add_group();
    _args.add_option("constraint_mask", "--constraint_mask", "Path to the constraint mask");
    _args.add_option("constraint_values", "--constraint_values", "Path to the constraint values");
    return _args.parse();
}

int RegularisationCommand::_execute(void)
{
    LOG(Info) << "Running regularization";

    float precision = _args.get<float>("precision", 0.5);
    int pyramid_levels = _args.get<int>("pyramid_levels", 6);

    LOG(Info) << "Precision: " << precision;
    LOG(Info) << "Pyramid levels: " << pyramid_levels;

    std::string constraint_mask_file = _args.get<std::string>("constraint_mask", "");
    std::string constraint_values_file = _args.get<std::string>("constraint_values", "");
    std::string output_file = _args.get<std::string>("output", "result_def.vtk");


    using namespace std::chrono;
    auto t_start = high_resolution_clock::now();

    LOG(Info) << "Input: '" << _args.positional("deformation") << "'";
    stk::VolumeFloat3 src = stk::read_volume(_args.positional("deformation").c_str());

    stk::Volume constraints_mask, constraints_values;
    if (!constraint_mask_file.empty()) {
        LOG(Info) << "Constraint mask: '" << constraint_mask_file << "'";
        constraints_mask = stk::read_volume(constraint_mask_file.c_str());
        if (!constraints_mask.valid()) return 1;
    } 
    if (!constraint_values_file.empty()) {
        LOG(Info) << "Constraint values: '" << constraint_values_file << "'";
        constraints_values = stk::read_volume(constraint_values_file.c_str());
        if (!constraints_values.valid()) return 1;
    }

    stk::VolumeFloat3 out = regularization(
        src,
        precision,
        pyramid_levels,
        constraints_mask,
        constraints_values
    );

    if (!out.valid()) {
        return 1;
    }

    auto t_end = high_resolution_clock::now();
    int elapsed = int(round(duration_cast<duration<double>>(t_end - t_start).count()));
    LOG(Info) << "Regularization completed in " << elapsed / 60 << ":" << std::right << std::setw(2) << std::setfill('0') << elapsed % 60;
    LOG(Info) << "Writing to '" << output_file << "'";
    stk::write_volume(output_file.c_str(), out);

    return EXIT_SUCCESS;
}
