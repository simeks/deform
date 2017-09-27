#include "config.h"
#include "config_file.h"
#include "cost_function.h"
#include "registration/blocked_graph_cut_optimizer.h"
#include "registration/transform.h"
#include "registration/volume_pyramid.h"

#include <framework/debug/assert.h>
#include <framework/debug/log.h>
#include <framework/filters/resample.h>
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

struct Args
{
    const char* param_file;
    
    const char* fixed_files[DF_MAX_IMAGE_PAIR_COUNT];
    const char* moving_files[DF_MAX_IMAGE_PAIR_COUNT];
};

struct Settings
{
    int pyramid_levels;
    int max_pyramid_level;

    float step_size;
    float regularization_weight;
   
    bool normalize_images;

    Settings() : 
        pyramid_levels(6),
        max_pyramid_level(0),
        step_size(0.5f),
        regularization_weight(0.05f),
        normalize_images(true)
    {
    }
};

struct RegistrationContext
{
    int _pyramid_levels; // Size of the multi-res pyramids
    int _pyramid_max_level; // Largest level to run (default: 0)
    int _image_pair_count; // Number of image pairs (e.g. fat, water and mask makes 3)

    float _step_size;
    float _regularization_weight;

    std::vector<VolumePyramid> _fixed_pyramids;
    std::vector<VolumePyramid> _moving_pyramids;
    VolumePyramid _deformation_pyramid;

    RegistrationContext() : 
        _pyramid_levels(-1),
        _pyramid_max_level(0),
        _image_pair_count(-1),
        _step_size(0),
        _regularization_weight(0) 
        {}
    ~RegistrationContext() {}
};


void print_help_and_exit(const char* err = 0)
{
    if (err)
        std::cout << "Error: " << err << std::endl;

    std::cout << "Arguments:" << std::endl
              << "-f<i> <file> : Filename of the i:th fixed image (i < " 
                << DF_MAX_IMAGE_PAIR_COUNT << ")*." << std::endl
              << "-m<i> <file> : Filename of the i:th moving image (i < " 
                << DF_MAX_IMAGE_PAIR_COUNT << ")*." << std::endl
              << "-p <file> : Filename of the parameter file (required)." << std::endl
              << "--help : Shows this help section." << std::endl
              << "*Requires a matching number of fixed and moving images";
    exit(0);
}
void parse_command_line(Args& args, int argc, char** argv)
{
    args = {0};

    /// Skip i=0 (name of executable)
    int i = 1;
    while (i < argc)
    {
        std::string token = argv[i];
        if (token[0] == '-')
        {
            int b = token[1] == '-' ? 2 : 1;
            std::string key = token.substr(b);

            if (key == "help")
            {
                print_help_and_exit();
            }
            else if (key == "p")
            {
                if (++i >= argc) 
                    print_help_and_exit("Missing arguments");
                args.param_file = argv[i];
            }
            else if (key[0] == 'f')
            {
                int img_index = std::stoi(key.substr(1));
                if (img_index >= DF_MAX_IMAGE_PAIR_COUNT)
                    print_help_and_exit();

                if (++i >= argc)
                    print_help_and_exit("Missing arguments");
                
                args.fixed_files[img_index] = argv[i];
            }
            else if (key[0] == 'm')
            {
                int img_index = std::stoi(key.substr(1));
                if (img_index >= DF_MAX_IMAGE_PAIR_COUNT)
                    print_help_and_exit();

                if (++i >= argc)
                    print_help_and_exit("Missing arguments");
                
                args.moving_files[img_index] = argv[i];
            }
            else
            {
                print_help_and_exit("Unrecognized option");
            }
        }
        else
        {
            print_help_and_exit("Unrecognized option");
        }
        ++i;
    }
}
/// Returns true if parsing was successful, false if not
void parse_parameter_file(Settings& settings, const char* file)
{
    // Assumes settings is filled with the default values beforehand

    ConfigFile cfg(file);

    if (cfg.keyExists("REGISTRATION_METHOD"))
    {
        LOG(Warning, "Parameter REGISTRATION_METHOD not applicable, ignoring.\n");
    }

    settings.pyramid_levels = cfg.getValueOfKey<int>("PYRAMID_LEVELS", settings.pyramid_levels);
    settings.max_pyramid_level = cfg.getValueOfKey<int>("MAX_RESOLUTION", settings.max_pyramid_level);
    settings.step_size = cfg.getValueOfKey<float>("STEPSIZE", settings.step_size);
    settings.regularization_weight = cfg.getValueOfKey<float>("REGULARIZATION_WEIGHT", settings.regularization_weight);
    settings.normalize_images = cfg.getValueOfKey<bool>("NORMALIZE_IMAGES", settings.normalize_images);
    
    LOG(Info, "Settings:\n");
    LOG(Info, "pyramid_levels = %d\n", settings.pyramid_levels);
    LOG(Info, "max_pyramid_level = %d\n", settings.max_pyramid_level);
    LOG(Info, "step_size = %f\n", settings.step_size);
    LOG(Info, "regularization_weight = %f\n", settings.regularization_weight);
    LOG(Info, "normalize_images = %d\n", settings.normalize_images);
}

// Identifies and loads the given file
// file : Filename
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

void initialize(RegistrationContext& ctx, const Settings& settings, int image_pair_count)
{
    ctx._pyramid_levels = settings.pyramid_levels;
    ctx._pyramid_max_level = settings.max_pyramid_level;
    ctx._image_pair_count = image_pair_count;

    ctx._step_size = settings.step_size;
    ctx._regularization_weight = settings.regularization_weight;

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

void upsample_and_save(RegistrationContext& ctx, int level)
{
    if (level == 0) return;

    int target_level = 0;
    int diff = level - target_level;
    assert(diff > 0);

    VolumeHelper<float3> def = ctx._deformation_pyramid.volume(target_level);
    VolumeHelper<float3> def_low = ctx._deformation_pyramid.volume(level);

    Dims dims = def.size();

    float factor = powf(0.5f, float(diff));
    
    #pragma omp parallel for
    for (int z = 0; z < int(dims.depth); ++z)
    {
        for (int y = 0; y < int(dims.height); ++y)
        {
            for (int x = 0; x < int(dims.width); ++x)
            {
                def(x, y, z) = (1.0f/factor) * def_low.linear_at(factor*x, factor*y, factor*z, volume::Border_Replicate);
            }
        }
    }

#ifdef DF_OUTPUT_DEBUG_VOLUMES
    std::stringstream ss;
    ss << "deformation_l" << level << ".vtk";
    vtk::write_volume(ss.str().c_str(), def);
    
    ss.str("");
    ss << "deformation_low_l" << level << ".vtk";
    vtk::write_volume(ss.str().c_str(), def_low);

    Volume moving = ctx._moving_pyramids[0].volume(0);

    ss.str("");
    ss << "transformed_l" << level << ".vtk";
    vtk::write_volume(ss.str().c_str(), transform_volume(moving, def));

    ss.str("");
    ss << "transformed_low_l" << level << ".vtk";
    vtk::write_volume(ss.str().c_str(), transform_volume(moving, def_low));
#endif
}


#ifdef DF_OUTPUT_DEBUG_VOLUMES
void save_volume_pyramid(RegistrationContext& ctx)
{
    for (int l = 0; l < ctx._pyramid_levels; ++l)
    {
        for (int i = 0; i < ctx._image_pair_count; ++i)
        {
            std::stringstream file;
            file << "fixed_pyramid_" << i << "_level_" << l << ".vtk";
            vtk::write_volume(file.str().c_str(), ctx._fixed_pyramids[i].volume(l));

            file.str("");
            file << "moving_pyramid_" << i << "_level_" << l << ".vtk";            
            vtk::write_volume(file.str().c_str(), ctx._moving_pyramids[i].volume(l));
        }
    }
}
#endif // DF_OUTPUT_DEBUG_VOLUMES


/// Runs the registration. 
/// Returns the resulting deformation field or an invalid volume if registration failed.
Volume execute_registration(RegistrationContext& ctx)
{
#ifdef DF_OUTPUT_DEBUG_VOLUMES
    save_volume_pyramid(ctx);
#endif
    
    // No copying of image data is performed here as Volume is simply a wrapper 
    std::vector<Volume> fixed_volumes(ctx._image_pair_count);
    std::vector<Volume> moving_volumes(ctx._image_pair_count);

    for (int l = ctx._pyramid_levels-1; l >= 0; --l)
    {
        VolumeFloat3 def = ctx._deformation_pyramid.volume(l);

        if (l >= ctx._pyramid_max_level)
        {
            LOG(Info, "Performing registration level %d\n", l);

            #if DF_DEBUG_LEVEL >= 1
                LOG(Debug, "[df%d] size: %d %d %d\n", l, def.size().width, def.size().height, def.size().depth);
                LOG(Debug, "[df%d] origin: %f %f %f\n", l, def.origin().x, def.origin().y, def.origin().z);
                LOG(Debug, "[df%d] spacing: %f %f %f\n", l, def.spacing().x, def.spacing().y, def.spacing().z);
            #endif
        
            for (int i = 0; i < ctx._image_pair_count; ++i)
            {
                fixed_volumes[i] = ctx._fixed_pyramids[i].volume(l);
                moving_volumes[i] = ctx._moving_pyramids[i].volume(l);
            }

            BlockedGraphCutOptimizer<EnergyFunction<double>, Regularizer> optimizer;
            EnergyFunction<double> unary_fn(1.0f - ctx._regularization_weight, fixed_volumes[0], moving_volumes[0]);
            Regularizer binary_fn(ctx._regularization_weight, fixed_volumes[0].spacing());

            // Calculate step size in voxels
            float3 fixed_spacing = fixed_volumes[0].spacing();
            float3 step_size_voxels{
                ctx._step_size / fixed_spacing.x,
                ctx._step_size / fixed_spacing.y,
                ctx._step_size / fixed_spacing.z
            };


#if DF_DEBUG_LEVEL >= 3
            LOG(Debug, "[f%d] spacing: %f, %f, %f\n", l, fixed_spacing.x, fixed_spacing.y, fixed_spacing.z);
            LOG(Debug, "step_size [voxels]: %f, %f, %f\n", step_size_voxels.x, step_size_voxels.y, step_size_voxels.z);
#endif
        
            optimizer.execute(unary_fn, binary_fn, step_size_voxels, def);
        }
        else
        {
            LOG(Info, "Skipping level %d\n", l);
        }
        
        if (l != 0)
        {
            Dims upsampled_dims = ctx._deformation_pyramid.volume(l - 1).size();
            ctx._deformation_pyramid.set_volume(l - 1,
                filters::upsample_vectorfield(def, upsampled_dims, ctx._deformation_pyramid.residual(l - 1)));
            
#ifdef DF_OUTPUT_DEBUG_VOLUMES
                upsample_and_save(ctx, l);
#endif // DF_OUTPUT_DEBUG_VOLUMES
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


int main(int argc, char* argv[])
{
    Args args = {0};
    parse_command_line(args, argc, argv);

    if (args.param_file == 0)
        print_help_and_exit();

    Settings settings;
    parse_parameter_file(settings, args.param_file);

    int image_pair_count = 0;
    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i)
    {
        if (args.fixed_files[i] && 
            args.moving_files[i] &&
            image_pair_count == i)
            ++image_pair_count;
    }

    if (image_pair_count == 0)
    {
        LOG(Error, "No (or invalid) input images, are you sure you (1) gave a matching \
            number of fixed and moving images, and (2) filled the slots incrementally (0, 1, ... n)?\n");
        return 1;
    }

    RegistrationContext ctx;
    initialize(ctx, settings, image_pair_count);

    VolumeDouble fixed_fat = load_volume("sandbox\\fixed_fat.vtk");
    if (!fixed_fat.valid()) return 1;
    VolumeDouble moving_fat = load_volume("sandbox\\moving_fat.vtk");
    if (!moving_fat.valid()) return 1;

#if DF_DEBUG_LEVEL >= 1
    {
        double min, max;
        fixed_fat.min_max(min, max);
        LOG(Info, "fixed_fat: min: %f, max: %f\n", min, max);
        moving_fat.min_max(min, max);
        LOG(Info, "moving_fat: min: %f, max: %f\n", min, max);
    }
#endif // DF_DEBUG_LEVEL >= 1

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
