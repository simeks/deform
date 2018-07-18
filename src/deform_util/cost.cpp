#include <deform_lib/arg_parser.h>
#include <deform_lib/filters/resample.h>
#include <deform_lib/registration/cost_function.h>
#include <deform_lib/registration/registration_engine.h>
#include <deform_lib/registration/settings.h>

#include <stk/common/assert.h>
#include <stk/common/log.h>
#include <stk/filters/normalize.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>

namespace {
    /// name : Name for printout
    bool validate_volume_properties(
        const stk::Volume& vol, 
        const dim3& expected_dims,
        const float3& expected_origin, 
        const float3& expected_spacing, 
        const std::string& name)
    {
        dim3 dims = vol.size();
        float3 origin = vol.origin();
        float3 spacing = vol.spacing();
        
        if (dims != expected_dims)
        {
            LOG(Error) << "Dimension mismatch for " << name << " (size: " 
                       << dims << ", expected: " << expected_dims << ")";
            return false;
        }

        // arbitrary epsilon but should suffice
        if (fabs(origin.x - expected_origin.x) > 0.0001f || 
            fabs(origin.y - expected_origin.y) > 0.0001f ||
            fabs(origin.z - expected_origin.z) > 0.0001f) 
        {
            LOG(Error) << "Origin mismatch for " << name 
                       << " (origin: " << origin << ", expected: " 
                       << expected_origin << ")";
            return false;
        }

        if (fabs(spacing.x - expected_spacing.x) > 0.0001f || 
            fabs(spacing.y - expected_spacing.y) > 0.0001f ||
            fabs(spacing.z - expected_spacing.z) > 0.0001f)
        {
            LOG(Error) << "Spacing mismatch for " << name 
                       << " (spacing: " << spacing << ", expected: " 
                       << expected_spacing << ")";
            return false;
        }
        return true;
    }
}

int run_cost(int argc, char* argv[])
{
    ArgParser args(argc, argv);
    args.add_positional("command", "registration, transform, regularize, jacobian");
    
    args.add_group();
    args.add_option("param_file",   "-p",           "Path to the parameter file");
    args.add_option("fixed{i}",     "-f{i}",        "Path to the i:th fixed image", true);
    args.add_option("moving{i}",    "-m{i}",        "Path to the i:th moving image", true);
    args.add_option("level", "-l", "Level in pyramid in which to compute the cost");
    args.add_option("unary_out", "--unary-out", "Output for unary term");
    args.add_option("binary_out", "--binary-out", "Output for binary term");
    args.add_group("Optional");
    args.add_option("init_deform",  "-d0", "Path to the initial deformation field");

    if (!args.parse()) {
        return 1;
    }
    
    std::string param_file = args.get<std::string>("param_file", "");

    Settings settings; // Default settings
    if (!param_file.empty()) {
        if (!parse_registration_file(param_file, settings))
            return 1;
    } 
    else {
        LOG(Info) << "Running with default settings.";
    }

    RegistrationEngine engine(settings);

    stk::Volume fixed_ref, moving_ref;
    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
        std::string fixed_id = "fixed" + std::to_string(i);
        std::string moving_id = "moving" + std::to_string(i);

        std::string fixed_file = args.get<std::string>(fixed_id, "");
        std::string moving_file = args.get<std::string>(moving_id, "");

        if (fixed_file == "" || moving_file == "")
            continue;

        stk::Volume fixed = stk::read_volume(fixed_file);
        if (!fixed.valid()) return 1;
        stk::Volume moving = stk::read_volume(moving_file);
        if (!moving.valid()) return 1;

        if (fixed.voxel_type() != moving.voxel_type()) {
            LOG(Error) << "Mismatch in voxel type between pairs at index " << i;
            return 1;
        }
        
        if (!fixed_ref.valid() || !moving_ref.valid()) {
            fixed_ref = fixed;
            moving_ref = moving;
        }
        else {
            if (!validate_volume_properties(fixed, fixed_ref.size(), 
                    fixed_ref.origin(), fixed_ref.spacing(), fixed_id)) {
                return 1;
            }
            if (!validate_volume_properties(moving, moving_ref.size(), 
                    moving_ref.origin(), moving_ref.spacing(), moving_id)) {
                return 1;
            }
        }

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

        engine.set_image_pair(i, fixed, moving, downsample_fn);

        LOG(Info) << "Fixed image [" << i << "]: '" << fixed_file << "'";
        LOG(Info) << "Moving image [" << i << "]: '" << moving_file << "'";
    }
    
    std::string init_deform_file = args.get<std::string>("init_deform", "");
    if (!init_deform_file.empty()) {
        stk::Volume initial_deformation = stk::read_volume(init_deform_file.c_str());
        if (!initial_deformation.valid()) return 1;

        if (!validate_volume_properties(initial_deformation, fixed_ref.size(), 
                fixed_ref.origin(), fixed_ref.spacing(), "initial deformation field"))
            return 1;

        engine.set_initial_deformation(initial_deformation);

        LOG(Info) << "Initial deformation '" << init_deform_file << "'";
    }

    int level = args.get<int>("level", 0);
    LOG(Info) << "Level: " << level;

    Regularizer binary_fn;
    engine.build_regularizer(level, binary_fn);

    UnaryFunction unary_fn;
    engine.build_unary_function(level, unary_fn);

    stk::VolumeFloat3 def = engine.deformation_field(level);

    stk::VolumeDouble unary_cost(def.size(), 0.0);
    unary_cost.set_origin(def.origin());
    unary_cost.set_spacing(def.spacing());
    
    stk::VolumeDouble3 binary_cost(def.size(), double3{0.0, 0.0, 0.0});
    binary_cost.set_origin(def.origin());
    binary_cost.set_spacing(def.spacing());

    dim3 dims = def.size();
    for (int z = 0; z < (int)dims.z; ++z) {
        for (int y = 0; y < (int)dims.y; ++y) {
            for (int x = 0; x < (int)dims.x; ++x) {
                float3 def1 = def(x,y,z);

                unary_cost(x,y,z) = unary_fn(int3{x,y,z}, def(x,y,z));

                double3 b{0,0,0};
                if (x + 1 < int(dims.x)) {
                    float3 def2 = def(x+1,y,z);
                    b.x = binary_fn({x,y,z}, def1, def2, {1,0,0});
                }
                if (y + 1 < int(dims.y)) {
                    float3 def2 = def(x,y+1,z);
                    b.y = binary_fn({x,y,z}, def1, def2, {0,1,0});
                }
                if (z + 1 < int(dims.z)) {
                    float3 def2 = def(x,y,z+1);
                    b.z = binary_fn({x,y,z}, def1, def2, {0,0,1});
                }

                binary_cost(x,y,z) = b;
            }
        }
    }

    std::string unary_out = args.get<std::string>("unary_out", "");
    std::string binary_out = args.get<std::string>("binary_out", "");

    if (!unary_out.empty()) 
        stk::write_volume(unary_out.c_str(), unary_cost);

    if (!binary_out.empty()) 
        stk::write_volume(binary_out.c_str(), binary_cost);

    return 0;
}
