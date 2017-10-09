#include "blocked_graph_cut_optimizer.h"
#include "cost_function.h"
#include "registration_engine.h"
#include "stats.h"
#include "transform.h"

#include <framework/debug/assert.h>
#include <framework/debug/log.h>
#include <framework/filters/resample.h>
#include <framework/volume/vtk.h>
#include <framework/volume/volume.h>
#include <framework/volume/volume_helper.h>

#include <sstream>
#include <string>

#ifdef DF_ENABLE_HARD_CONSTRAINTS
#include "hard_constraints.h"
#endif // DF_ENABLE_HARD_CONSTRAINTS


namespace
{
    /// name : Name for printout
    bool validate_volume_properties(
        const Volume& vol, 
        const Dims& expected_dims,
        const float3& expected_origin, 
        const float3& expected_spacing, 
        const char* name)
    {
        Dims dims = vol.size();
        float3 origin = vol.origin();
        float3 spacing = vol.spacing();
     
        if (dims != expected_dims)
        {
            LOG(Error, "Dimension mismatch for %s (size: %d %d %d, expected: %d %d %d)\n", 
                name,
                dims.width, dims.height, dims.depth,
                expected_dims.width, expected_dims.height, expected_dims.depth);
            return false;
        }

        if (fabs(origin.x - expected_origin.x) > 0.0001f || 
            fabs(origin.y - expected_origin.y) > 0.0001f ||
            fabs(origin.z - expected_origin.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Origin mismatch for %s (origin: %f %f %f, expected: %f %f %f)\n",
                        name,
                        origin.x, origin.y, origin.z,
                        expected_origin.x, expected_origin.y, expected_origin.z);
            return false;
        }

        if (fabs(spacing.x - expected_spacing.x) > 0.0001f || 
            fabs(spacing.y - expected_spacing.y) > 0.0001f ||
            fabs(spacing.z - expected_spacing.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Spacing mismatch for %s (spacing: %f %f %f, expected: %f %f %f)\n",
                        name,
                        spacing.x, spacing.y, spacing.z,
                        expected_spacing.x, expected_spacing.y, expected_spacing.z);
            return false;
        }
        return true;
    }

#ifdef DF_ENABLE_HARD_CONSTRAINTS
    void constrain_deformation_field(VolumeFloat3& def, const VolumeUInt8& mask, const VolumeFloat3& values)
    {
        assert(def.size() == mask.size() && def.size() == values.size());
        Dims dims = def.size();
        for (int z = 0; z < int(dims.depth); ++z)
        {
            for (int y = 0; y < int(dims.height); ++y)
            {
                for (int x = 0; x < int(dims.width); ++x)
                {
                    if (mask(x, y, z) > 0)
                    {
                        def(x, y, z) = values(x, y, z);
                    }
                }
            }
        }
    }
#endif // DF_ENABLE_HARD_CONSTRAINTS
}

RegistrationEngine::RegistrationEngine(const Settings& settings) :
    _pyramid_levels(settings.pyramid_levels),
    _pyramid_max_level(settings.max_pyramid_level),
    _step_size(settings.step_size),
    _regularization_weight(settings.regularization_weight),
    _image_pair_count(0)
{
}
RegistrationEngine::~RegistrationEngine()
{
}


void RegistrationEngine::initialize(int image_pair_count)
{
    _image_pair_count = image_pair_count;
    _fixed_pyramids.resize(_image_pair_count);
    _moving_pyramids.resize(_image_pair_count);

    for (int i = 0; i < _image_pair_count; ++i)
    {
        _fixed_pyramids[i].set_level_count(_pyramid_levels);
        _moving_pyramids[i].set_level_count(_pyramid_levels);
    }
    _deformation_pyramid.set_level_count(_pyramid_levels);

    #ifdef DF_ENABLE_HARD_CONSTRAINTS
        _constraints_pyramid.set_level_count(_pyramid_levels);
        _constraints_mask_pyramid.set_level_count(_pyramid_levels);
    #endif // DF_ENABLE_HARD_CONSTRAINTS
}
void RegistrationEngine::set_initial_deformation(const Volume& def)
{
    assert(def.voxel_type() == voxel::Type_Float3); // Only single-precision supported for now
    assert(_pyramid_levels);

    _deformation_pyramid.build_from_base_with_residual(def, filters::downsample_vectorfield);
}
void RegistrationEngine::set_image_pair(
    int i, 
    const Volume& fixed, 
    const Volume& moving,
    Volume (*downsample_fn)(const Volume&, float))
{
    assert(i < DF_MAX_IMAGE_PAIR_COUNT);

    _fixed_pyramids[i].build_from_base(fixed, downsample_fn);
    _moving_pyramids[i].build_from_base(moving, downsample_fn);
}
#ifdef DF_ENABLE_HARD_CONSTRAINTS
void RegistrationEngine::set_hard_constraints(const VolumeUInt8& mask, const VolumeFloat3& values)
{
    _constraints_mask_pyramid.set_volume(0, mask);
    _constraints_pyramid.set_volume(0, values);
    for (int i = 0; i < _pyramid_levels-1; ++i)
    {
        VolumeUInt8 prev_mask = _constraints_mask_pyramid.volume(i);
        VolumeFloat3 prev_values = _constraints_pyramid.volume(i);

        _constraints_mask_pyramid.set_volume(i+1, 
            hard_constraints::downsample_mask_by_2(prev_mask));
        _constraints_pyramid.set_volume(i+1, 
            hard_constraints::downsample_values_by_2(prev_mask, prev_values));
    }
}
#endif // DF_ENABLE_HARD_CONSTRAINTS

Volume RegistrationEngine::execute()
{
    if (_image_pair_count == 0)
    {
        LOG(Error, "Nothing to register\n");
        return Volume();
    }

    if (!_deformation_pyramid.volume(0).valid())
    {
        // No initial deformation, create a field with all zeros
        
        Volume base = _fixed_pyramids[0].volume(0);

        VolumeFloat3 initial(base.size(), float3{0, 0, 0});
        initial.set_origin(base.origin());
        initial.set_spacing(base.spacing());
        set_initial_deformation(initial);
    }

    #ifdef DF_OUTPUT_DEBUG_VOLUMES
        save_volume_pyramid();
    #endif

    // No copying of image data is performed here as Volume is simply a wrapper 
    std::vector<Volume> fixed_volumes(_image_pair_count);
    std::vector<Volume> moving_volumes(_image_pair_count);

    for (int l = _pyramid_levels-1; l >= 0; --l)
    {
        STATS_RESET("Stat_Energy");

        VolumeFloat3 def = _deformation_pyramid.volume(l);

        if (l >= _pyramid_max_level)
        {
            LOG(Info, "Performing registration level %d\n", l);

            #if DF_DEBUG_LEVEL >= 1
                LOG(Debug, "[df%d] size: %d %d %d\n", l, def.size().width, def.size().height, def.size().depth);
                LOG(Debug, "[df%d] origin: %f %f %f\n", l, def.origin().x, def.origin().y, def.origin().z);
                LOG(Debug, "[df%d] spacing: %f %f %f\n", l, def.spacing().x, def.spacing().y, def.spacing().z);
            #endif
        
            for (int i = 0; i < _image_pair_count; ++i)
            {
                fixed_volumes[i] = _fixed_pyramids[i].volume(l);
                moving_volumes[i] = _moving_pyramids[i].volume(l);
            }

            #ifdef DF_ENABLE_HARD_CONSTRAINTS
                BlockedGraphCutOptimizer<
                    EnergyFunctionWithConstraints<float>, 
                    Regularizer> optimizer;

                EnergyFunctionWithConstraints<float> unary_fn(
                    1.0f - _regularization_weight, 
                    fixed_volumes[0], 
                    moving_volumes[0],
                    _constraints_mask_pyramid.volume(l),
                    _constraints_pyramid.volume(l)
                );

                // Fix constrained voxels by updating the initial deformation field
                constrain_deformation_field(
                    def, 
                    _constraints_mask_pyramid.volume(l),
                    _constraints_pyramid.volume(l)
                );

            #else
                BlockedGraphCutOptimizer<
                    EnergyFunction<float>,
                    Regularizer> optimizer;
                
                    EnergyFunction<float> unary_fn(
                        1.0f - _regularization_weight, 
                        fixed_volumes[0], 
                        moving_volumes[0]
                    );
            #endif
            
            Regularizer binary_fn(_regularization_weight, fixed_volumes[0].spacing());

            // Calculate step size in voxels
            float3 fixed_spacing = fixed_volumes[0].spacing();
            float3 step_size_voxels{
                _step_size / fixed_spacing.x,
                _step_size / fixed_spacing.y,
                _step_size / fixed_spacing.z
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
            Dims upsampled_dims = _deformation_pyramid.volume(l - 1).size();
            _deformation_pyramid.set_volume(l - 1,
                filters::upsample_vectorfield(def, upsampled_dims, _deformation_pyramid.residual(l - 1)));
                        
            #ifdef DF_OUTPUT_DEBUG_VOLUMES
                upsample_and_save(l);
            #endif // DF_OUTPUT_DEBUG_VOLUMES
        }
        else
        {
            _deformation_pyramid.set_volume(0, def);
        }

        #ifdef DF_ENABLE_STATS
            std::stringstream ss;
            ss << "stat_energy_level_" << l << ".txt";
            std::string energy_log = ss.str();

            STATS_DUMP("Stat_Energy", energy_log.c_str());
        #endif // DF_ENABLE_STATS
    }

    return _deformation_pyramid.volume(0);
}

bool RegistrationEngine::validate_input()
{
    // Rules:
    // * All volumes for the same subject (i.e. fixed or moving) must have the same dimensions
    // * All volumes for the same subject (i.e. fixed or moving) need to have the same origin and spacing
    // * For simplicity any given initial deformation field must match the fixed image properties (size, origin, spacing)
    // If hard constraints are enabled:
    // * Constraint mask and values must match fixed image


    Dims fixed_dims = _fixed_pyramids[0].volume(0).size();
    Dims moving_dims = _moving_pyramids[0].volume(0).size();

    float3 fixed_origin = _fixed_pyramids[0].volume(0).origin();
    float3 moving_origin = _moving_pyramids[0].volume(0).origin();
    
    float3 fixed_spacing = _fixed_pyramids[0].volume(0).spacing();
    float3 moving_spacing = _moving_pyramids[0].volume(0).spacing();

    for (int i = 1; i < _image_pair_count; ++i)
    {
        if (!_fixed_pyramids[i].volume(0).valid() ||
            !_moving_pyramids[i].volume(0).valid())
        {
            LOG(Error, "Missing image(s) at index %d\n", i);
            return false;
        }

        std::stringstream ss; ss << "fixed image id " << i;
        std::string fixed_name = ss.str();

        if (!validate_volume_properties(_fixed_pyramids[i].volume(0), 
                fixed_dims, fixed_origin, fixed_spacing, fixed_name.c_str()))
            return false;
    
        ss.str(""); 
        ss << "moving image id " << i;
        std::string moving_name = ss.str();

        if (!validate_volume_properties(_moving_pyramids[i].volume(0), 
                moving_dims, moving_origin, moving_spacing, moving_name.c_str()))
            return false;
    }

    if (_deformation_pyramid.volume(0).valid() && !validate_volume_properties(_deformation_pyramid.volume(0), 
            fixed_dims, fixed_origin, fixed_spacing, "initial deformation field"))
        return false;

    #ifdef DF_ENABLE_HARD_CONSTRAINTS
        if (_constraints_pyramid.volume(0).valid() && !validate_volume_properties(_constraints_pyramid.volume(0), 
                fixed_dims, fixed_origin, fixed_spacing, "constraints values"))
            return false;

        if (_constraints_mask_pyramid.volume(0).valid() && !validate_volume_properties(_constraints_mask_pyramid.volume(0), 
                fixed_dims, fixed_origin, fixed_spacing, "constraints mask"))
            return false;

    #endif // DF_ENABLE_HARD_CONSTRAINTS

    return true;
}
#ifdef DF_OUTPUT_DEBUG_VOLUMES
void RegistrationEngine::upsample_and_save(int level)
{
    if (level == 0) return;

    int target_level = 0;
    int diff = level - target_level;
    assert(diff > 0);

    VolumeHelper<float3> def = _deformation_pyramid.volume(target_level);
    VolumeHelper<float3> def_low = _deformation_pyramid.volume(level);

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

    std::stringstream ss;
    ss << "deformation_l" << level << ".vtk";
    std::string tmp = ss.str();
    vtk::write_volume(tmp.c_str(), def);
    
    ss.str("");
    ss << "deformation_low_l" << level << ".vtk";
    tmp = ss.str();
    vtk::write_volume(tmp.c_str(), def_low);

    Volume moving = _moving_pyramids[0].volume(0);

    ss.str("");
    ss << "transformed_l" << level << ".vtk";
    tmp = ss.str();
    vtk::write_volume(tmp.c_str(), transform_volume(moving, def));

    ss.str("");
    ss << "transformed_low_l" << level << ".vtk";
    tmp = ss.str();
    vtk::write_volume(tmp.c_str(), transform_volume(moving, def_low));
}
void RegistrationEngine::save_volume_pyramid()
{
    for (int l = 0; l < _pyramid_levels; ++l)
    {
        for (int i = 0; i < _image_pair_count; ++i)
        {
            std::stringstream file;
            file << "fixed_pyramid_" << i << "_level_" << l << ".vtk";
            vtk::write_volume(file.str().c_str(), _fixed_pyramids[i].volume(l));

            file.str("");
            file << "moving_pyramid_" << i << "_level_" << l << ".vtk";            
            vtk::write_volume(file.str().c_str(), _moving_pyramids[i].volume(l));
        }

        std::stringstream file;
        file << "initial_deformation_level_" << l << ".vtk";
        vtk::write_volume(file.str().c_str(), _deformation_pyramid.volume(l));
    }
}
#endif // DF_OUTPUT_DEBUG_VOLUMES
