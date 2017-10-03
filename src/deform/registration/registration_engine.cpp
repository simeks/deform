#include "blocked_graph_cut_optimizer.h"
#include "cost_function.h"
#include "registration_engine.h"
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
namespace
{
    
}
#endif // DF_ENABLE_HARD_CONSTRAINTS


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
    _deformation_pyramid.build_from_base()
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

            BlockedGraphCutOptimizer<EnergyFunction<double>, Regularizer> optimizer;
            EnergyFunction<double> unary_fn(1.0f - _regularization_weight, fixed_volumes[0], moving_volumes[0]);
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
    }

    return _deformation_pyramid.volume(0);
}

bool RegistrationEngine::validate_input()
{
    // Rules:
    // * All volumes for the same subject (i.e. fixed or moving) must have the same dimensions
    // * All volumes for the same subject (i.e. fixed or moving) need to have the same origin and spacing
    // * For simplicity any given initial deformation field must match the fixed image properties (size, origin, spacing)
    
    Dims fixed_dims = _fixed_pyramids[0].volume(0).size();
    Dims moving_dims = _moving_pyramids[0].volume(0).size();

    float3 fixed_origin = _fixed_pyramids[0].volume(0).origin();
    float3 moving_origin = _moving_pyramids[0].volume(0).origin();
    
    float3 fixed_spacing = _fixed_pyramids[0].volume(0).spacing();
    float3 moving_spacing = _moving_pyramids[0].volume(0).spacing();

    for (int i = 1; i < _image_pair_count; ++i)
    {
        Dims fixed_dims_i = _fixed_pyramids[i].volume(0).size();
        if (fixed_dims_i != fixed_dims)
        {
            LOG(Error, "Dimension mismatch for fixed image id %d (size: %d %d %d, expected: %d %d %d)\n", i, 
                fixed_dims_i.width, fixed_dims_i.height, fixed_dims_i.depth,
                fixed_dims.width, fixed_dims.height, fixed_dims.depth);
            return false;
        }
        Dims moving_dims_i = _moving_pyramids[i].volume(0).size();
        if (moving_dims_i != moving_dims)
        {
            LOG(Error, "Dimension mismatch for moving image id %d (size: %d %d %d, expected: %d %d %d)\n", i, 
                moving_dims_i.width, moving_dims_i.height, moving_dims_i.depth,
                moving_dims.width, moving_dims.height, moving_dims.depth);
            return false;
        }
        
        float3 fixed_origin_i = _fixed_pyramids[i].volume(0).origin();
        if (fabs(fixed_origin_i.x - fixed_origin.x) > 0.0001f || 
            fabs(fixed_origin_i.y - fixed_origin.y) > 0.0001f ||
            fabs(fixed_origin_i.z - fixed_origin.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Origin mismatch for fixed image id %d (origin: %f %f %f, expected: %f %f %f)\n", i, 
                        fixed_origin_i.x, fixed_origin_i.y, fixed_origin_i.z,
                        fixed_origin.x, fixed_origin.y, fixed_origin.z);
            return false;
        }

        float3 fixed_spacing_i = _fixed_pyramids[i].volume(0).spacing();
        if (fabs(fixed_spacing_i.x - fixed_spacing.x) > 0.0001f || 
            fabs(fixed_spacing_i.y - fixed_spacing.y) > 0.0001f ||
            fabs(fixed_spacing_i.z - fixed_spacing.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Spacing mismatch for fixed image id %d (spacing: %f %f %f, expected: %f %f %f)\n", i, 
                        fixed_spacing_i.x, fixed_spacing_i.y, fixed_spacing_i.z,
                        fixed_spacing.x, fixed_spacing.y, fixed_spacing.z);
            return false;
        }
        
        float3 moving_origin_i = _moving_pyramids[i].volume(0).origin();
        if (fabs(moving_origin_i.x - moving_origin.x) > 0.0001f || 
            fabs(moving_origin_i.y - moving_origin.y) > 0.0001f ||
            fabs(moving_origin_i.z - moving_origin.z) > 0.0001f) // arbitrary epsilon but should suffice
        {
            LOG(Error, "Origin mismatch for moving image id %d (origin: %f %f %f, expected: %f %f %f)\n", i, 
                        moving_origin_i.x, moving_origin_i.y, moving_origin_i.z,
                        moving_origin.x, moving_origin.y, moving_origin.z);
            return false;
        }

        float3 moving_spacing_i = _moving_pyramids[i].volume(0).spacing();
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

    const Volume& initial_def = _deformation_pyramid.volume(0);
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
    vtk::write_volume(ss.str().c_str(), def);
    
    ss.str("");
    ss << "deformation_low_l" << level << ".vtk";
    vtk::write_volume(ss.str().c_str(), def_low);

    Volume moving = _moving_pyramids[0].volume(0);

    ss.str("");
    ss << "transformed_l" << level << ".vtk";
    vtk::write_volume(ss.str().c_str(), transform_volume(moving, def));

    ss.str("");
    ss << "transformed_low_l" << level << ".vtk";
    vtk::write_volume(ss.str().c_str(), transform_volume(moving, def_low));
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
