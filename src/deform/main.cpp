#include "config_file.h"
#include "registration/blocked_graph_cut_optimizer.h"
#include "registration/transform.h"
#include "registration/volume_pyramid.h"

#include <framework/debug/assert.h>
#include <framework/debug/log.h>
#include <framework/filters/resample.h>
#include <framework/platform/arg_parser.h>
#include <framework/platform/file_path.h>
#include <framework/volume/volume.h>
#include <framework/volume/volume_helper.h>
#include <framework/volume/stb.h>
#include <framework/volume/vtk.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace settings
{
    float step_size = 0.5f; // [mm]
    bool output_all_levels = true; // Outputs deformation fields and deformed volumes for all levels in pyramid
}

struct RegistrationContext
{
    int _pyramid_levels; // Size of the multi-res pyramids
    int _image_pair_count; // Number of image pairs (e.g. fat, water and mask makes 3)

    std::vector<VolumePyramid> _fixed_pyramids;
    std::vector<VolumePyramid> _moving_pyramids;
    VolumePyramid _deformation_pyramid;

    RegistrationContext() : _pyramid_levels(-1), _image_pair_count(-1) {}
    ~RegistrationContext() {}
};


// Identifies and loads the given file
// file : Filename
// is_2d [out] : Indicates if the volume is a 2d volume
// Returns the loaded volume, if load failed the returned volume will be flagged as invalid 
Volume load_volume(const std::string& file)
{
    FilePath path(file);
    std::string ext = path.extension();
    
    // To lower case
    std::transform(ext.begin(), ext.end(), ext.begin(), [](char c){ return (char)::tolower(c); });

    if (ext == "vtk")
    {
        vtk::Reader reader;
        Volume vol = reader.execute(file.c_str());
        if (!vol.valid())        
        {
            LOG(Error, "Failed to read image: %s\n", reader.last_error());
        }
        return vol;
    }
    else if (ext == "png")
    {
        std::string err;
        Volume vol = stb::read_image(file.c_str());
        if (!vol.valid())
        {
            LOG(Error, "Failed to read image: %s\n", stb::last_read_error());
        }
        return vol;
    }
    else
    {
        LOG(Error, "Unsupported file extension: '%s'\n", ext.c_str());
    }
    // Returning an "invalid" volume
    return Volume();
}

void initialize(RegistrationContext& ctx, int pyramid_levels, int image_pair_count)
{
    ctx._pyramid_levels = pyramid_levels;
    ctx._image_pair_count = image_pair_count;

    ctx._fixed_pyramids.resize(ctx._image_pair_count);
    ctx._moving_pyramids.resize(ctx._image_pair_count);

    for (int i = 0; i < ctx._image_pair_count; ++i)
    {
        ctx._fixed_pyramids[i].set_level_count(ctx._pyramid_levels);
        ctx._moving_pyramids[i].set_level_count(ctx._pyramid_levels);
    }
    ctx._deformation_pyramid.set_level_count(ctx._pyramid_levels);
}

/// Validates all volumes and makes sure everything is in order.
/// Should be called before performing executing the registration.
/// Returns true if the validation was successful, false if not.
bool validate_input(RegistrationContext& ctx)
{
    // Rules:
    // * All volumes for the same subject (i.e. fixed or moving) must have the same dimensions
    // * All volumes for the same subject (i.e. fixed or moving) need to have the same origin and spacing
    // * For simplicity any given initial deformation field must match the fixed image properties (size, origin, spacing)
    
    Dims fixed_dims = ctx._fixed_pyramids[0].volume(0).size();
    Dims moving_dims = ctx._moving_pyramids[0].volume(0).size();

    float3 fixed_origin = ctx._fixed_pyramids[0].volume(0).origin();
    float3 moving_origin = ctx._moving_pyramids[0].volume(0).origin();
    
    float3 fixed_spacing = ctx._fixed_pyramids[0].volume(0).spacing();
    float3 moving_spacing = ctx._moving_pyramids[0].volume(0).spacing();

    for (int i = 1; i < ctx._image_pair_count; ++i)
    {
        Dims fixed_dims_i = ctx._fixed_pyramids[i].volume(0).size();
        if (fixed_dims_i != fixed_dims)
        {
            LOG(Error, "Dimension mismatch for fixed image id %d (size: %d %d %d, expected: %d %d %d)\n", i, 
                fixed_dims_i.width, fixed_dims_i.height, fixed_dims_i.depth,
                fixed_dims.width, fixed_dims.height, fixed_dims.depth);
            return false;
        }
        Dims moving_dims_i = ctx._moving_pyramids[i].volume(0).size();
        if (moving_dims_i != moving_dims)
        {
            LOG(Error, "Dimension mismatch for moving image id %d (size: %d %d %d, expected: %d %d %d)\n", i, 
                moving_dims_i.width, moving_dims_i.height, moving_dims_i.depth,
                moving_dims.width, moving_dims.height, moving_dims.depth);
            return false;
        }
        
        float3 fixed_origin_i = ctx._fixed_pyramids[i].volume(0).origin();
        if (fabs(fixed_origin_i.x - fixed_origin.x) > 0.0001f || 
            fabs(fixed_origin_i.y - fixed_origin.y) > 0.0001f ||
            fabs(fixed_origin_i.z - fixed_origin.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Origin mismatch for fixed image id %d (origin: %f %f %f, expected: %f %f %f)\n", i, 
                        fixed_origin_i.x, fixed_origin_i.y, fixed_origin_i.z,
                        fixed_origin.x, fixed_origin.y, fixed_origin.z);
            return false;
        }

        float3 fixed_spacing_i = ctx._fixed_pyramids[i].volume(0).spacing();
        if (fabs(fixed_spacing_i.x - fixed_spacing.x) > 0.0001f || 
            fabs(fixed_spacing_i.y - fixed_spacing.y) > 0.0001f ||
            fabs(fixed_spacing_i.z - fixed_spacing.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Spacing mismatch for fixed image id %d (spacing: %f %f %f, expected: %f %f %f)\n", i, 
                        fixed_spacing_i.x, fixed_spacing_i.y, fixed_spacing_i.z,
                        fixed_spacing.x, fixed_spacing.y, fixed_spacing.z);
            return false;
        }
        
        float3 moving_origin_i = ctx._moving_pyramids[i].volume(0).origin();
        if (fabs(moving_origin_i.x - moving_origin.x) > 0.0001f || 
            fabs(moving_origin_i.y - moving_origin.y) > 0.0001f ||
            fabs(moving_origin_i.z - moving_origin.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Origin mismatch for moving image id %d (origin: %f %f %f, expected: %f %f %f)\n", i, 
                        moving_origin_i.x, moving_origin_i.y, moving_origin_i.z,
                        moving_origin.x, moving_origin.y, moving_origin.z);
            return false;
        }

        float3 moving_spacing_i = ctx._moving_pyramids[i].volume(0).spacing();
        if (fabs(moving_spacing_i.x - moving_spacing.x) > 0.0001f || 
            fabs(moving_spacing_i.y - moving_spacing.y) > 0.0001f ||
            fabs(moving_spacing_i.z - moving_spacing.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Spacing mismatch for moving image id %d (spacing: %f %f %f, expected: %f %f %f)\n", i, 
                        moving_spacing_i.x, moving_spacing_i.y, moving_spacing_i.z,
                        moving_spacing.x, moving_spacing.y, moving_spacing.z);
            return false;
        }
    }

    const Volume& initial_def = ctx._deformation_pyramid.volume(0);
    if (initial_def.valid())
    {
        Dims def_dims = initial_def.size();
        float3 def_origin = initial_def.origin();
        float3 def_spacing = initial_def.spacing();
     
        if (def_dims != fixed_dims)
        {
            LOG(Error, "Dimension mismatch for initial deformation field (size: %d %d %d, expected: %d %d %d)\n", 
                def_dims.width, def_dims.height, def_dims.depth,
                fixed_dims.width, fixed_dims.height, fixed_dims.depth);
            return false;
        }

        if (fabs(def_origin.x - fixed_origin.x) > 0.0001f || 
            fabs(def_origin.y - fixed_origin.y) > 0.0001f ||
            fabs(def_origin.z - fixed_origin.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Origin mismatch for initial deformation field (origin: %f %f %f, expected: %f %f %f)\n",
                        def_origin.x, def_origin.y, def_origin.z,
                        fixed_origin.x, fixed_origin.y, fixed_origin.z);
            return false;
        }

        if (fabs(def_spacing.x - fixed_spacing.x) > 0.0001f || 
            fabs(def_spacing.y - fixed_spacing.y) > 0.0001f ||
            fabs(def_spacing.z - fixed_spacing.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Spacing mismatch for initial deformation field (spacing: %f %f %f, expected: %f %f %f)\n",
                        def_spacing.x, def_spacing.y, def_spacing.z,
                        fixed_spacing.x, fixed_spacing.y, fixed_spacing.z);
            return false;
        }
    }

    return true;
}

/// Runs the registration. 
/// Returns the resulting deformation field or an invalid volume if registration failed.
Volume execute_registration(RegistrationContext& ctx)
{
    // No copying of image data is performed here as Volume is simply a wrapper 
    std::vector<Volume> fixed_volumes(ctx._image_pair_count);
    std::vector<Volume> moving_volumes(ctx._image_pair_count);

    for (int l = ctx._pyramid_levels-1; l >= 0; --l)
    {
        for (int i = 0; i < ctx._image_pair_count; ++i)
        {
            fixed_volumes[i] = ctx._fixed_pyramids[i].volume(l);
            moving_volumes[i] = ctx._moving_pyramids[i].volume(l);
        }

        VolumeFloat3 def = ctx._deformation_pyramid.volume(l);

        BlockedGraphCutOptimizer<EnergyFunction<double>, Regularizer> optimizer;
        EnergyFunction<double> unary_fn(fixed_volumes[0], moving_volumes[0]);
        Regularizer binary_fn(fixed_volumes[0].spacing());

        // Calculate step size in voxels
        float3 fixed_spacing = fixed_volumes[0].spacing();
        float3 step_size_voxels{
            settings::step_size / fixed_spacing.x,
            settings::step_size / fixed_spacing.y,
            settings::step_size / fixed_spacing.z
        };
        optimizer.execute(unary_fn, binary_fn, step_size_voxels, def);

        if (l != 0)
        {
            Dims upsampled_dims = ctx._deformation_pyramid.volume(l - 1).size();
            ctx._deformation_pyramid.set_volume(l - 1,
                filters::upsample_vectorfield(def, upsampled_dims, ctx._deformation_pyramid.residual(l - 1)));
            
            if (settings::output_all_levels)
            {
                
            }
        }
        else
        {
            ctx._deformation_pyramid.set_volume(0, def);
        }
    }

    return ctx._deformation_pyramid.volume(0);
}

void set_initial_deformation(RegistrationContext& ctx, const Volume& def)
{
    assert(def.voxel_type() == voxel::Type_Float3); // Only single-precision supported for now
    assert(ctx._pyramid_levels);

    ctx._deformation_pyramid.build_from_base_with_residual(def, filters::downsample_vectorfield);
}

void set_image_pair(
    RegistrationContext& ctx,
    int i, 
    const Volume& fixed, 
    const Volume& moving,
    Volume (*downsample_fn)(const Volume&, float))
{
    assert(i < ctx._image_pair_count);

    ctx._fixed_pyramids[i].build_from_base(fixed, downsample_fn);
    ctx._moving_pyramids[i].build_from_base(moving, downsample_fn);
}

void print_help()
{
    std::cout   << "Arguments:" << std::endl
                << "-p=<filename> : parameter file (obligatory)" << std::endl
                << "-f<i>=<filename> : Filename for the i:th fixed image" << std::endl
                << "-m<i>=<filename> : Filename for the i:th moving image" << std::endl
                << "-h, --help : Show this help section" << std::endl;

}


int main(int argc, char* argv[])
{
    ArgParser args(argc, argv);

    // if (args.is_set("help") || args.is_set("h"))
    // {
    //     print_help();
    //     return 1;
    // }

    // if (args.num_tokens() < 1)
    // {
    //     print_help();
    //     return 1;
    // }

    RegistrationContext ctx;
    initialize(ctx, 6, 1);

    auto fixed_fat = load_volume("sandbox\\fixed_fat.vtk");
    if (!fixed_fat.valid()) return 1;
    auto moving_fat = load_volume("sandbox\\moving_fat.vtk");
    if (!moving_fat.valid()) return 1;
    
    set_image_pair(ctx, 0, fixed_fat, moving_fat, filters::downsample_volume_gaussian);
    
    VolumeFloat3 starting_guess(fixed_fat.size(), float3{0, 0, 0});
    starting_guess.set_origin(fixed_fat.origin());
    starting_guess.set_spacing(fixed_fat.spacing());
    set_initial_deformation(ctx, starting_guess);

    validate_input(ctx);

    Volume def = execute_registration(ctx);
    vtk::write_volume("sandbox\\result_def.vtk", def);

    Volume result = transform_volume(moving_fat, def);
    vtk::write_volume("sandbox\\result.vtk", result);

    // vtk::Reader reader;
    // Volume vol = reader.execute("C:\\data\\test.vtk");//args.token(0).c_str());
    // if (reader.failed())
    // {
    //     std::cout << reader.last_error();
    //     return 1;
    // }

    // std::string fi, mi, fixed_file, moving_file;
    // for (int i = 0; ; ++i)
    // {
    //     std::stringstream ss;
    //     ss << "-f" << i;
    //     fi = ss.str();

    //     ss.str("");
    //     ss << "-m" << i;
    //     mi = ss.str();

    //     if (args.is_set(fi) && args.is_set(mi))
    //     {
    //         // engine.set_fixed_image(i, file);
    //         // engine.set_moving_image(i, file);
    //     }
    // }

    return 0;
}