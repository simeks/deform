#include "registration_engine.h"

#include <framework/debug/assert.h>
#include <framework/filters/resample.h>
#include <framework/volume/volume.h>

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
Volume RegistrationEngine::run()
{
    return Volume();
}
