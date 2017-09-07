#include "optimizer.h"
#include "registration_engine.h"

#include <framework/debug/assert.h>
#include <framework/debug/log.h>
#include <framework/filters/resample.h>
#include <framework/volume/volume.h>

#include <math.h>

RegistrationEngine::RegistrationEngine(int pyramid_levels, int image_pair_count) :
    _pyramid_levels(pyramid_levels),
    _image_pair_count(image_pair_count)
{
    _fixed_pyramids.resize(_image_pair_count);
    _moving_pyramids.resize(_image_pair_count);
}
RegistrationEngine::~RegistrationEngine()
{
}
void RegistrationEngine::set_initial_deformation(const Volume& def)
{
    assert(def.voxel_type() == voxel::Type_Float3); // Only single-precision supported for now
    assert(_pyramid_levels);

    _deformation_pyramid.build_from_base_with_residual(def, filters::downsample_vectorfield);
}
void RegistrationEngine::set_image_pair(int i, const ImagePairDesc& desc, const Volume& fixed, const Volume& moving)
{
    _fixed_pyramids[i].build_from_base(fixed, desc.downsample_fn);
    _moving_pyramids[i].build_from_base(moving, desc.downsample_fn);
}
Volume RegistrationEngine::run(Optimizer& optimizer)
{
    if (!validate_volumes())
        return Volume();

    // No copying of image data is performed here as Volume is simply a wrapper 
    std::vector<Volume> fixed_volumes(_image_pair_count);
    std::vector<Volume> moving_volumes(_image_pair_count);

    for (int l = _pyramid_levels-1; l >= 0; --l)
    {
        for (int i = 0; i < _image_pair_count; ++i)
        {
            fixed_volumes[i] = _fixed_pyramids[i].volume(l);
            moving_volumes[i] = _fixed_pyramids[i].volume(l);
        }

        Volume def = _deformation_pyramid.volume(l);

        optimizer.execute(
            fixed_volumes.data(), 
            moving_volumes.data(), 
            _image_pair_count,
            def
        );

        def = filters::upsample_vectorfield(def, 2.0f, _deformation_pyramid.residual(l-1));
        _deformation_pyramid.set_volume(l-1, def);
    }

    return Volume();
}
bool RegistrationEngine::validate_volumes()
{
    // Rules:
    // * All volumes for the same subject (i.e. fixed or moving) must have the same dimensions
    // * All volumes for the same subject (i.e. fixed or moving) need to have the same origin and spacing
    
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
            LOG(Error, "Spacing mismatch for fixed image id %d (origin: %f %f %f, expected: %f %f %f)\n", i, 
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
            LOG(Error, "Spacing mismatch for moving image id %d (origin: %f %f %f, expected: %f %f %f)\n", i, 
                        moving_spacing_i.x, moving_spacing_i.y, moving_spacing_i.z,
                        moving_spacing.x, moving_spacing.y, moving_spacing.z);
            return false;
        }
    }

    return true;
}
