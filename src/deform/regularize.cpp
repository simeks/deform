#include <stk/io/io.h>
#include <stk/filters/gaussian_filter.h>

#include <deform_lib/regularize.h>
#include <deform_lib/filters/resample.h>
#include <deform_lib/registration/volume_pyramid.h>
#include <deform_lib/registration/voxel_constraints.h>

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

    VolumePyramid deformation_pyramid;
    deformation_pyramid.set_level_count(pyramid_levels);

    {
        LOG(Info) << "Input: '" << _args.positional("deformation") << "'";

        stk::Volume src = stk::read_volume(_args.positional("deformation").c_str());
        if (!src.valid()) return 1;

        if (src.voxel_type() != stk::Type_Float3) {
            LOG(Error) << "Invalid voxel type for deformation field, expected float3";
            return 1;
        }

    #ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
        deformation_pyramid.build_from_base_with_residual(src, filters::downsample_vectorfield_by_2);
    #else
        deformation_pyramid.build_from_base(src, filters::downsample_vectorfield_by_2);
    #endif
    }

    bool use_constraints = false;
    stk::Volume constraints_mask, constraints_values;
    if (!constraint_mask_file.empty() && !constraint_values_file.empty()) {
        LOG(Info) << "Constraint mask: '" << constraint_mask_file << "'";
        LOG(Info) << "Constraint values: '" << constraint_values_file << "'";

        constraints_mask = stk::read_volume(constraint_mask_file.c_str());
        if (!constraints_mask.valid()) return 1;

        constraints_values = stk::read_volume(constraint_values_file.c_str());
        if (!constraints_values.valid()) return 1;

        use_constraints = true;
    }
    else {
        constraints_mask = stk::VolumeUChar(deformation_pyramid.volume(0).size(), uint8_t{0});
        constraints_values = stk::VolumeFloat3(deformation_pyramid.volume(0).size(), float3{0, 0, 0});
    }

    VolumePyramid constraints_mask_pyramid, constraints_pyramid;
    voxel_constraints::build_pyramids(
        constraints_mask,
        constraints_values,
        pyramid_levels,
        constraints_mask_pyramid,
        constraints_pyramid
    );

    // Initialization is only needed if we have constraints
    if (use_constraints) {
        // Perform initialization at the coarsest resolution
        stk::VolumeFloat3 def = deformation_pyramid.volume(pyramid_levels-1);
        initialize_regularization(
            def,
            constraints_mask_pyramid.volume(pyramid_levels-1),
            constraints_pyramid.volume(pyramid_levels-1)
        );
    }

    for (int l = pyramid_levels-1; l >= 0; --l) {
        stk::VolumeFloat3 def = deformation_pyramid.volume(l);

        LOG(Info) << "Performing regularization level " <<  l;

        do_regularization(
            def,
            constraints_mask_pyramid.volume(l),
            constraints_pyramid.volume(l),
            precision
        );

        if (l != 0) {
            dim3 upsampled_dims = deformation_pyramid.volume(l - 1).size();
            deformation_pyramid.set_volume(l - 1,
        #ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
                filters::upsample_vectorfield(def, upsampled_dims, deformation_pyramid.residual(l - 1))
        #else
                filters::upsample_vectorfield(def, upsampled_dims)
        #endif
            );
        }
        else {
            deformation_pyramid.set_volume(0, def);
        }
    }
    auto t_end = high_resolution_clock::now();
    int elapsed = int(round(duration_cast<duration<double>>(t_end - t_start).count()));
    LOG(Info) << "Regularization completed in " << elapsed / 60 << ":" << std::right << std::setw(2) << std::setfill('0') << elapsed % 60;
    LOG(Info) << "Writing to '" << output_file << "'";
    stk::write_volume(output_file.c_str(), deformation_pyramid.volume(0));

    return 0;
}
