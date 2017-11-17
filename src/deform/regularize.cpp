#include "registration/volume_pyramid.h"
#include "registration/voxel_constraints.h"

#include <framework/debug/log.h>
#include <framework/filters/resample.h>
#include <framework/math/float3.h>
#include <framework/math/int3.h>
#include <framework/platform/timer.h>
#include <framework/volume/volume_helper.h>
#include <framework/volume/vtk.h>
#include <iostream>

// From main
Volume load_volume(const std::string& file);

namespace
{
    float default_precision = 0.5f;
    int default_pyramid_levels = 6;

    int3 neighbors[] = {
        {1, 0, 0},
        {-1, 0, 0},
        {0, 1, 0},
        {0, -1, 0},
        {0, 0, 1},
        {0, 0, -1}
    };
}

void print_help_and_exit(const char* exe, const char* err=NULL)
{
    if (err) std::cout << "Error: " << err << std::endl;
    std::cout << "Usage: " << exe << " regularize <deformation field>" << std::endl;
    std::cout << "Arguments: " << std::endl
              << "-p : Precision (Default: " << default_precision << ")" << std::endl
              << "-l : Number of pyramid levels (Default: " << default_pyramid_levels << ")" << std::endl
              << "-o, --output : Output file" << std::endl;
              
    exit(1);
}

void initialize_regularization(
    VolumeFloat3& def, 
    const VolumeUInt8& constraints_mask,
    const VolumeFloat3& constraints_values
)
{
    float3 spacing = def.spacing();
    float3 inv_spacing {
        1.0f / spacing.x,
        1.0f / spacing.y,
        1.0f / spacing.z
    };

    float neighbor_weight[6];
    for (int i = 0; i < 6; ++i)
    {
        float3 n = {
            inv_spacing.x * neighbors[i].x,
            inv_spacing.y * neighbors[i].y,
            inv_spacing.z * neighbors[i].z
        };
        neighbor_weight[i] = math::length_squared(n);
    }

    Dims dims = def.size();
    VolumeUInt8 visited(dims, 0);

    size_t nvisited = 0;
    for (int z = 0; z < int(dims.depth); ++z)
    {
        for (int y = 0; y < int(dims.height); ++y)
        {
            for (int x = 0; x < int(dims.width); ++x)
            {
                if (constraints_mask(x, y, z) > 0)
                {
                    visited(x, y, z) = 1;
                    def(x, y, z) = constraints_values(x, y, z);
                    ++nvisited;
                }
            }
        }
    }

    size_t nelems = dims.width * dims.height * dims.depth;
    while (nvisited < nelems)
    {
        for (int z = 0; z < int(dims.depth); ++z)
        {
            for (int y = 0; y < int(dims.height); ++y)
            {
                for (int x = 0; x < int(dims.width); ++x)
                {
                    int3 p{x, y, z};

                    if (constraints_mask(p) > 0)
                    {
                        def(p) = constraints_values(p);
                        continue;
                    }
                    
                    float3 new_def{0};
                    float3 old_def = def(p);

                    float weight_sum = 0;

                    for (int i = 0; i < 6; ++i)
                    {
                        if (visited.at(p + neighbors[i], volume::Border_Replicate) > 0)
                        {
                            if (visited(p) == 0)
                            {
                                ++nvisited;
                                visited(p) = 1;
                            }

                            weight_sum += neighbor_weight[i];
                            new_def = new_def + neighbor_weight[i] * def.at(p + neighbors[i], volume::Border_Replicate);

                        }
                    }
                    if (weight_sum > 0)
                    {
                        def(p) = new_def / weight_sum;
                    }
                }
            }
        }
   }
}

void do_regularization(
    VolumeFloat3& def, 
    const VolumeUInt8& constraints_mask,
    const VolumeFloat3& constraints_values,
    float precision
)
{
    float3 spacing = def.spacing();
    float3 inv_spacing {
        1.0f / spacing.x,
        1.0f / spacing.y,
        1.0f / spacing.z
    };

    Dims dims = def.size();

    float neighbor_weight[6];
    for (int i = 0; i < 6; ++i)
    {
        float3 n = {
            inv_spacing.x * neighbors[i].x,
            inv_spacing.y * neighbors[i].y,
            inv_spacing.z * neighbors[i].z
        };
        neighbor_weight[i] = math::length_squared(n);
    }

    bool done = false;
    while (!done)
    {
        done = true;

        for (int black_or_red = 0; black_or_red < 2; ++black_or_red)
        {
            #pragma omp parallel for
            for (int z = 0; z < int(dims.depth); ++z)
            {
                for (int y = 0; y < int(dims.height); ++y)
                {
                    for (int x = 0; x < int(dims.width); ++x)
                    {
                        int3 p {x, y, z};

                        int off = (z) % 2;
                        off = (y + off) % 2;
                        off = (x + off) % 2;
    
                        if (off == black_or_red) continue;

                        if (constraints_mask(p) > 0)
                        {
                            def(p) = constraints_values(p);
                            continue;
                        }

                        float3 new_def{0};
                        float3 old_def = def(p);
                        float weight_sum = 0.0f;

                        for (int i = 0; i < 6; ++i)
                        {
                            weight_sum += neighbor_weight[i];
                            new_def = new_def + neighbor_weight[i] 
                                * def.at(p + neighbors[i], volume::Border_Replicate);
                        }

                        new_def = new_def / weight_sum;
                        //Successive over relaxation, relaxation factor=1.5
                        new_def = old_def + 1.5f*(new_def - old_def);

                        def(p) = new_def;
                        float diff = math::length(new_def - old_def);
                        if (diff > precision)
                        {
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
    if (argc < 3)
        print_help_and_exit(argv[0]);

    float precision = default_precision;
    int pyramid_levels = default_pyramid_levels;

    const char* constraint_mask_file = NULL;
    const char* constraint_values_file = NULL;
    const char* output_file = "result_def.vtk";
        
    /// Skip i=0,1,3 (name of executable + "regularize" + "<deformation field>")
    int i = 3;
    while (i < argc)
    {
        std::string token = argv[i];
        if (token[0] == '-')
        {
            int b = token[1] == '-' ? 2 : 1;
            std::string key = token.substr(b);

            if (key == "help")
            {
                print_help_and_exit(argv[0]);
            }
            else if (key == "p")
            {
                if (++i >= argc) 
                    print_help_and_exit(argv[0], "Missing arguments");
                precision = (float)atof(argv[i]);
            }
            else if (key == "l")
            {
                if (++i >= argc) 
                    print_help_and_exit(argv[0], "Missing arguments");
                pyramid_levels = atoi(argv[i]);
            }
            else if (key == "constraint_mask" ||
                     key == "constraints_mask")
            {
                if (++i >= argc) 
                    print_help_and_exit(argv[0], "Missing arguments");
                constraint_mask_file = argv[i];
            }
            else if (key == "constraint_values" ||
                     key == "constraints_values")
            {
                if (++i >= argc) 
                    print_help_and_exit(argv[0], "Missing arguments");
                constraint_values_file = argv[i];
            }
            else if (key == "o" || key == "output")
            {
                if (++i >= argc)
                    print_help_and_exit(argv[0], "Missing arguments");
                output_file = argv[i];
            }
            else
            {
                print_help_and_exit(argv[0], "Unrecognized option");
            }
        }
        else
        {
            print_help_and_exit(argv[0], "Unrecognized option");
        }
        ++i;
    }

    double t_start = timer::seconds();

    VolumePyramid deformation_pyramid;
    deformation_pyramid.set_level_count(pyramid_levels);
    
    {
        Volume src = load_volume(argv[2]);
        if (!src.valid()) return 1;
            
        if (src.voxel_type() != voxel::Type_Float3)
        {
            LOG(Error, "Invalid voxel type for deformation field, expected float3\n");
            return 1;
        }
        
        deformation_pyramid.build_from_base_with_residual(src, filters::downsample_vectorfield);
    }

    bool use_constraints = false;
    Volume constraints_mask, constraints_values;
    if (constraint_mask_file && constraint_values_file)
    {
        constraints_mask = load_volume(constraint_mask_file);
        if (!constraints_mask.valid()) return 1;

        constraints_values = load_volume(constraint_values_file);
        if (!constraints_values.valid()) return 1;
    
        use_constraints = true;
    }
    else
    {
        constraints_mask = VolumeUInt8(deformation_pyramid.volume(0).size(), 0);
        constraints_values = VolumeFloat3(deformation_pyramid.volume(0).size(), float3{0});
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
    if (use_constraints)
    {
        // Perform initialization at the coarsest resolution
        VolumeFloat3 def = deformation_pyramid.volume(pyramid_levels-1);
        initialize_regularization(
            def,
            constraints_mask_pyramid.volume(pyramid_levels-1),
            constraints_pyramid.volume(pyramid_levels-1)
        );
    }
    
    for (int l = pyramid_levels-1; l >= 0; --l)
    {
        VolumeFloat3 def = deformation_pyramid.volume(l);
        
        LOG(Info, "Performing regularization level %d\n", l);
        
        do_regularization(
            def,
            constraints_mask_pyramid.volume(l),
            constraints_pyramid.volume(l),
            precision
        );

        if (l != 0)
        {
            Dims upsampled_dims = deformation_pyramid.volume(l - 1).size();
            deformation_pyramid.set_volume(l - 1,
                filters::upsample_vectorfield(def, upsampled_dims, deformation_pyramid.residual(l - 1)));
        }
        else
        {
            deformation_pyramid.set_volume(0, def);
        }
    }
    double t_end = timer::seconds();
    int elapsed = int(round(t_end - t_start));
    LOG(Info, "Regularization completed in %d:%02d\n", elapsed / 60, elapsed % 60);

    vtk::write_volume(output_file, deformation_pyramid.volume(0));

    return 0;
}
