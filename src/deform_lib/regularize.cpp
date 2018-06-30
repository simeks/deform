#include "arg_parser.h"
#include "filters/resample.h"
#include "platform/timer.h"
#include "registration/volume_pyramid.h"
#include "registration/voxel_constraints.h"

#include <stk/common/log.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>

#include <iomanip>
#include <iostream>


namespace
{
    int3 neighbors[] = {
        {1, 0, 0},
        {-1, 0, 0},
        {0, 1, 0},
        {0, -1, 0},
        {0, 0, 1},
        {0, 0, -1}
    };
}

void initialize_regularization(
    stk::VolumeFloat3& def, 
    const stk::VolumeUChar& constraints_mask,
    const stk::VolumeFloat3& constraints_values
)
{
    float3 spacing = def.spacing();
    float3 inv_spacing {
        1.0f / spacing.x,
        1.0f / spacing.y,
        1.0f / spacing.z
    };

    float neighbor_weight[6];
    for (int i = 0; i < 6; ++i) {
        float3 n = {
            inv_spacing.x * neighbors[i].x,
            inv_spacing.y * neighbors[i].y,
            inv_spacing.z * neighbors[i].z
        };
        neighbor_weight[i] = stk::norm2(n);
    }

    dim3 dims = def.size();
    stk::VolumeUChar visited(dims, uint8_t{0});

    size_t nvisited = 0;
    for (int z = 0; z < int(dims.z); ++z) {
        for (int y = 0; y < int(dims.y); ++y) {
            for (int x = 0; x < int(dims.x); ++x) {
                if (constraints_mask(x, y, z) > 0) {
                    visited(x, y, z) = 1;
                    def(x, y, z) = constraints_values(x, y, z);
                    ++nvisited;
                }
            }
        }
    }

    size_t nelems = dims.x * dims.y * dims.z;
    while (nvisited < nelems) {
        for (int z = 0; z < int(dims.z); ++z) {
            for (int y = 0; y < int(dims.y); ++y) {
                for (int x = 0; x < int(dims.x); ++x) {
                    int3 p{x, y, z};

                    if (constraints_mask(p) > 0) {
                        def(p) = constraints_values(p);
                        continue;
                    }
                    
                    float3 new_def{0};

                    float weight_sum = 0;

                    for (int i = 0; i < 6; ++i) {
                        if (visited.at(p + neighbors[i], stk::Border_Replicate) > 0) {
                            if (visited(p) == 0) {
                                ++nvisited;
                                visited(p) = 1;
                            }

                            weight_sum += neighbor_weight[i];
                            new_def = new_def + neighbor_weight[i] * def.at(p + neighbors[i], stk::Border_Replicate);
                        }
                    }
                    if (weight_sum > 0) {
                        def(p) = new_def / weight_sum;
                    }
                }
            }
        }
   }
}

void do_regularization(
    stk::VolumeFloat3& def, 
    const stk::VolumeUChar& constraints_mask,
    const stk::VolumeFloat3& constraints_values,
    float precision
)
{
    float3 spacing = def.spacing();
    float3 inv_spacing {
        1.0f / spacing.x,
        1.0f / spacing.y,
        1.0f / spacing.z
    };

    dim3 dims = def.size();

    float neighbor_weight[6];
    for (int i = 0; i < 6; ++i) {
        float3 n = {
            inv_spacing.x * neighbors[i].x,
            inv_spacing.y * neighbors[i].y,
            inv_spacing.z * neighbors[i].z
        };
        neighbor_weight[i] = stk::norm2(n);
    }

    bool done = false;
    while (!done) {
        done = true;

        for (int black_or_red = 0; black_or_red < 2; ++black_or_red) {
            #pragma omp parallel for
            for (int z = 0; z < int(dims.z); ++z) {
                for (int y = 0; y < int(dims.y); ++y) {
                    for (int x = 0; x < int(dims.x); ++x) {
                        int3 p {x, y, z};

                        int off = (z) % 2;
                        off = (y + off) % 2;
                        off = (x + off) % 2;
    
                        if (off == black_or_red) continue;

                        if (constraints_mask(p) > 0) {
                            def(p) = constraints_values(p);
                            continue;
                        }

                        float3 new_def{0};
                        float3 old_def = def(p);
                        float weight_sum = 0.0f;

                        for (int i = 0; i < 6; ++i) {
                            weight_sum += neighbor_weight[i];
                            new_def = new_def + neighbor_weight[i] 
                                * def.at(p + neighbors[i], stk::Border_Replicate);
                        }

                        new_def = new_def / weight_sum;
                        //Successive over relaxation, relaxation factor=1.5
                        new_def = old_def + 1.5f*(new_def - old_def);

                        def(p) = new_def;
                        float diff = stk::norm(new_def - old_def);
                        if (diff > precision) {
                            done = false;
                        }
                    }
                }
            }

        }
    }
}

int run_regularize(int argc, char* argv[])
{
    ArgParser args(argc, argv);
    args.add_positional("command", "registration, transform, regularize, jacobian");
    args.add_positional("deformation", "Path to the deformation field to regularize");
    args.add_group();
    args.add_option("output", "-o, --output", "Path to output (default: result_def.vtk)");
    args.add_option("precision", "-p, --precision", "Precision (default: 0.5)");
    args.add_option("pyramid_levels", "-l, --levels", "Number of pyramid levels (default: 6)");
    args.add_group();
    args.add_option("constraint_mask", "--constraint_mask", "Path to the constraint mask");
    args.add_option("constraint_values", "--constraint_values", "Path to the constraint values");

    if (!args.parse()) {
        return 1;
    }

    float precision = args.get<float>("precision", 0.5);
    int pyramid_levels = args.get<int>("pyramid_levels", 6);

    std::string constraint_mask_file = args.get<std::string>("constraint_mask", "");
    std::string constraint_values_file = args.get<std::string>("constraint_values", "");
    std::string output_file = args.get<std::string>("output", "result_def.vtk");

    double t_start = timer::seconds();

    VolumePyramid deformation_pyramid;
    deformation_pyramid.set_level_count(pyramid_levels);
    
    {
        stk::Volume src = stk::read_volume(args.positional("deformation").c_str());
        if (!src.valid()) return 1;
            
        if (src.voxel_type() != stk::Type_Float3) {
            LOG(Error) << "Invalid voxel type for deformation field, expected float3";
            return 1;
        }
        
        deformation_pyramid.build_from_base_with_residual(src, filters::downsample_vectorfield);
    }

    bool use_constraints = false;
    stk::Volume constraints_mask, constraints_values;
    if (!constraint_mask_file.empty() && !constraint_values_file.empty()) {
        constraints_mask = stk::read_volume(constraint_mask_file.c_str());
        if (!constraints_mask.valid()) return 1;

        constraints_values = stk::read_volume(constraint_values_file.c_str());
        if (!constraints_values.valid()) return 1;
    
        use_constraints = true;
    }
    else {
        constraints_mask = stk::VolumeUChar(deformation_pyramid.volume(0).size(), uint8_t{0});
        constraints_values = stk::VolumeFloat3(deformation_pyramid.volume(0).size(), float3{0});
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
                filters::upsample_vectorfield(def, upsampled_dims, deformation_pyramid.residual(l - 1)));
        }
        else {
            deformation_pyramid.set_volume(0, def);
        }
    }
    double t_end = timer::seconds();
    int elapsed = int(round(t_end - t_start));
    LOG(Info) << "Regularization completed in " << elapsed / 60 << ":" << std::setw(2) << std::setfill('0') << elapsed % 60;

    stk::write_volume(output_file.c_str(), deformation_pyramid.volume(0));

    return 0;
}
