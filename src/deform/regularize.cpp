#include "registration/volume_pyramid.h"
#include "regularize.h"

#include <framework/math/float3.h>
#include <framework/volume/volume.h>
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

void print_help_and_exit(const char* exe, const char* err)
{
    if (err) std::cout << "Error: " << err << std::endl;
    std::cout << "Usage: " << exe << " regularize <deformation field>" << std::endl;
    std::cout << "Arguments: " << std::endl
              << "-p : Precision (Default: " << default_precision << ")" << std::endl
              << "-l : Number of pyramid levels (Default: " << default_pyramid_levels << ")" << std::endl;
              
    exit(1);
}

void do_regularization(
    VolumeFloat3& def, 
    VolumeUInt8& constraints_mask,
    VolumeFloat3& constraints_values,
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
        neighbor_weight[i] = float3::length_squared(n);
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

                        if (constraint_mask(p) > 0)
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
                            new_def = new_def + def.at(p + neightbors[i], volume::Border_Replicate);
                        }

                        new_def = new_def / weight_sum;
                        //Successive over relaxation, relaxation factor=1.5
                        new_def = old_def + 1.5f*(new_def - old_def);

                        def(p) = new_def;
                        float diff = float3::length_squared(new_def - old_def);
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
        print_usage(argv[0]);

    float precision = default_precision;
    int pyramid_levels = default_pyramid_levels;
        
    /// Skip i=0,1 (name of executable + "regularize")
    int i = 2;
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

    Volume src = load_volume(argv[2]);
    if (!src.valid()) return 1;
    
    if (!src.voxel_type() != voxel::Type_Float3)
    {
        LOG(Error, "Invalid voxel type for deformation field, expected float3\n");
        return 1;
    }

    VolumePyramid src_pyramid;
    src_pyramid.set_level_count(pyramid_levels);
    src_pyramid.build_from_base_with_residual(df, filters::downsample_vectorfield);

    
    for (int l = pyramid_levels-1; l >= 0; --l)
    {
        VolumeFloat3 def = src_pyramid.volume(l);
        
        LOG(Info, "Performing regularization level %d\n", l);
        
        do_regularization();
        
    }

}
