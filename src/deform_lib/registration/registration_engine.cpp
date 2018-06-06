#include "../cost_function.h"
#include "../filters/resample.h"

#include "blocked_graph_cut_optimizer.h"
#include "registration_engine.h"
#include "transform.h"

#include <stk/common/assert.h>
#include <stk/common/log.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>

#include <sstream>
#include <string>

#include "voxel_constraints.h"


namespace
{
    float origin_epsilon = 0.1f; // .1 mm

    void constrain_deformation_field(stk::VolumeFloat3& def, 
        const stk::VolumeUChar& mask, const stk::VolumeFloat3& values)
    {
        ASSERT(def.size() == mask.size() && def.size() == values.size());
        dim3 dims = def.size();
        for (int z = 0; z < int(dims.z); ++z) {
            for (int y = 0; y < int(dims.y); ++y) {
                for (int x = 0; x < int(dims.x); ++x) {
                    if (mask(x, y, z) > 0) {
                        def(x, y, z) = values(x, y, z);
                    }
                }
            }
        }
    }
}

RegistrationEngine::RegistrationEngine(const Settings& settings) :
    _settings(settings)
{
    _fixed_pyramids.resize(DF_MAX_IMAGE_PAIR_COUNT);
    _moving_pyramids.resize(DF_MAX_IMAGE_PAIR_COUNT);

    _deformation_pyramid.set_level_count(_settings.num_pyramid_levels);

    #ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        _regularization_weight_map.set_level_count(_settings.num_pyramid_levels);
    #endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    _constraints_pyramid.set_level_count(_settings.num_pyramid_levels);
    _constraints_mask_pyramid.set_level_count(_settings.num_pyramid_levels);
}
RegistrationEngine::~RegistrationEngine()
{
}
void RegistrationEngine::set_initial_deformation(const stk::Volume& def)
{
    ASSERT(def.voxel_type() == stk::Type_Float3); // Only single-precision supported for now
    ASSERT(_settings.num_pyramid_levels);

    _deformation_pyramid.build_from_base_with_residual(def, filters::downsample_vectorfield);
}
void RegistrationEngine::set_image_pair(
    int i, 
    const stk::Volume& fixed, 
    const stk::Volume& moving,
    stk::Volume (*downsample_fn)(const stk::Volume&, float))
{
    ASSERT(i < DF_MAX_IMAGE_PAIR_COUNT);
    
    _fixed_pyramids[i].set_level_count(_settings.num_pyramid_levels);
    _moving_pyramids[i].set_level_count(_settings.num_pyramid_levels);
    
    _fixed_pyramids[i].build_from_base(fixed, downsample_fn);
    _moving_pyramids[i].build_from_base(moving, downsample_fn);
}
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
void RegistrationEngine::set_regularization_weight_map(const stk::Volume& map)
{
    _regularization_weight_map.build_from_base(map, filters::downsample_volume_gaussian);
}
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

void RegistrationEngine::set_voxel_constraints(const stk::VolumeUChar& mask, const stk::VolumeFloat3& values)
{
    voxel_constraints::build_pyramids(
        mask, 
        values,
        _settings.num_pyramid_levels,
        _constraints_mask_pyramid,
        _constraints_pyramid
    );
}

stk::Volume RegistrationEngine::execute()
{
    if (!_deformation_pyramid.volume(0).valid()) {
        // No initial deformation, create a field with all zeros
        
        stk::Volume base = _fixed_pyramids[0].volume(0);

        stk::VolumeFloat3 initial(base.size(), float3{0, 0, 0});
        initial.set_origin(base.origin());
        initial.set_spacing(base.spacing());
        set_initial_deformation(initial);
    }

    #ifdef DF_OUTPUT_DEBUG_VOLUMES
        save_volume_pyramid();
    #endif

    // No copying of image data is performed here as Volume is simply a wrapper 
    stk::Volume fixed_volumes[DF_MAX_IMAGE_PAIR_COUNT];
    stk::Volume moving_volumes[DF_MAX_IMAGE_PAIR_COUNT];

    for (int l = _settings.num_pyramid_levels-1; l >= 0; --l) {
        stk::VolumeFloat3 def = _deformation_pyramid.volume(l);

        if (l >= _settings.pyramid_stop_level) {
            LOG(Info) << "Performing registration level " << l;

            for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
                if (_fixed_pyramids[i].levels() > 0)
                    fixed_volumes[i] = _fixed_pyramids[i].volume(l);
                if (_moving_pyramids[i].levels() > 0)
                    moving_volumes[i] = _moving_pyramids[i].volume(l);
            }

            UnaryFunction unary_fn(_settings.regularization_weight);
            if (_constraints_mask_pyramid.volume(l).valid()) {
                unary_fn.add_function(
                    std::make_unique<SoftConstraintsFunction>(
                        _constraints_mask_pyramid.volume(l),
                        _constraints_pyramid.volume(l),
                        _settings.constraints_weight
                    )
                );
            }

            for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
                if (!fixed_volumes[i].valid() || !moving_volumes[i].valid())
                    continue; // Skip empty slots
                
                auto& slot = _settings.image_slots[i];
                if (slot.cost_function == Settings::ImageSlot::CostFunction_SSD) {
                    if (fixed_volumes[i].voxel_type() == stk::Type_Float) {
                        unary_fn.add_function(
                            std::make_unique<SquaredDistanceFunction<float>>(
                                fixed_volumes[i],
                                moving_volumes[i]
                            )
                        );
                    }
                    else if (fixed_volumes[i].voxel_type() == stk::Type_Double) {
                        unary_fn.add_function(
                            std::make_unique<SquaredDistanceFunction<double>>(
                                fixed_volumes[i],
                                moving_volumes[i]
                            )
                        );
                    }
                    else {
                        LOG(Error) << "Invalid cost function for volume of type " << fixed_volumes[i].voxel_type();
                        return stk::Volume();
                    }
                }
                else if (slot.cost_function == Settings::ImageSlot::CostFunction_NCC)
                {
                    if (fixed_volumes[i].voxel_type() == stk::Type_Float)
                    {
                        unary_fn.add_function(
                            std::make_unique<NCCFunction<float>>(
                                fixed_volumes[i],
                                moving_volumes[i]
                            )
                        );
                    }
                    else if (fixed_volumes[i].voxel_type() == stk::Type_Double)
                    {
                        unary_fn.add_function(
                            std::make_unique<NCCFunction<double>>(
                                fixed_volumes[i],
                                moving_volumes[i]
                            )
                        );
                    }
                    else
                    {
                        LOG(Error) << "Invalid cost function for volume of type " << fixed_volumes[i].voxel_type();
                        return stk::Volume();
                    }
                }
                
            }
            BlockedGraphCutOptimizer<UnaryFunction, Regularizer> optimizer(
                _settings.block_size,
                _settings.block_energy_epsilon
            );
  

            if (_constraints_mask_pyramid.volume(l).valid())
            {
                // Fix constrained voxels by updating the initial deformation field
                constrain_deformation_field(
                    def,
                    _constraints_mask_pyramid.volume(l),
                    _constraints_pyramid.volume(l)
                );
            }
            
            Regularizer binary_fn(_settings.regularization_weight, fixed_volumes[0].spacing());

            #ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
                if (_regularization_weight_map.volume(l).valid())
                    binary_fn.set_weight_map(_regularization_weight_map.volume(l));
            #endif
    
            float3 fixed_spacing = fixed_volumes[0].spacing();
            for (int sm = 1; sm > 0; --sm) {
                // Calculate step size in voxels
                float3 step_size_voxels {
                    sm*_settings.step_size / fixed_spacing.x,
                    sm*_settings.step_size / fixed_spacing.y,
                    sm*_settings.step_size / fixed_spacing.z
                };

                DLOG(Info) << "step_size: " << sm*_settings.step_size << " [mm] => "
                           << step_size_voxels << " [voxels]";

                optimizer.execute(unary_fn, binary_fn, step_size_voxels, def);
            }
        }
        else {
            LOG(Info) << "Skipping level " << l;
        }
        
        if (l != 0) {
            dim3 upsampled_dims = _deformation_pyramid.volume(l - 1).size();
            _deformation_pyramid.set_volume(l - 1,
                filters::upsample_vectorfield(def, upsampled_dims, _deformation_pyramid.residual(l - 1)));
                        
            #ifdef DF_OUTPUT_DEBUG_VOLUMES
                upsample_and_save(l);
            #endif // DF_OUTPUT_DEBUG_VOLUMES
        }
        else {
            _deformation_pyramid.set_volume(0, def);
        }
    }

    return _deformation_pyramid.volume(0);
}
#ifdef DF_OUTPUT_DEBUG_VOLUMES
void RegistrationEngine::upsample_and_save(int level)
{
    if (level == 0) return;

    int target_level = 0;
    int diff = level - target_level;
    assert(diff > 0);

    stk::VolumeHelper<float3> def = _deformation_pyramid.volume(target_level);
    stk::VolumeHelper<float3> def_low = _deformation_pyramid.volume(level);

    dim3 dims = def.size();

    float factor = powf(0.5f, float(diff));
    
    #pragma omp parallel for
    for (int z = 0; z < int(dims.z); ++z) {
        for (int y = 0; y < int(dims.y); ++y) {
            for (int x = 0; x < int(dims.x); ++x) {
                def(x, y, z) = (1.0f/factor) * def_low.linear_at(factor*x, factor*y, factor*z, volume::Border_Replicate);
            }
        }
    }

    std::stringstream ss;
    ss << "deformation_l" << level << ".vtk";
    std::string tmp = ss.str();
    stk::write_volume(tmp.c_str(), def);
    
    ss.str("");
    ss << "deformation_low_l" << level << ".vtk";
    tmp = ss.str();
    stk::write_volume(tmp.c_str(), def_low);

    stk::Volume moving = _moving_pyramids[0].volume(0);

    ss.str("");
    ss << "transformed_l" << level << ".vtk";
    tmp = ss.str();
    stk::write_volume(tmp.c_str(), transform_volume(moving, def));

    ss.str("");
    ss << "transformed_low_l" << level << ".vtk";
    tmp = ss.str();
    stk::write_volume(tmp.c_str(), transform_volume(moving, def_low));
}
void RegistrationEngine::save_volume_pyramid()
{
    for (int l = 0; l < _settings.num_pyramid_levels; ++l) {
        for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {

            if (_fixed_pyramids[i].levels() > 0) {
                std::stringstream file;
                file << "fixed_pyramid_" << i << "_level_" << l << ".vtk";
                stk::write_volume(file.str().c_str(), _fixed_pyramids[i].volume(l));
            }

            if (_moving_pyramids[i].levels() > 0) {
                std::stringstream file;
                file << "moving_pyramid_" << i << "_level_" << l << ".vtk";            
                stk::write_volume(file.str().c_str(), _moving_pyramids[i].volume(l));
            }
        }

        std::stringstream file;
        file << "initial_deformation_level_" << l << ".vtk";
        stk::write_volume(file.str().c_str(), _deformation_pyramid.volume(l));
    }
}
#endif // DF_OUTPUT_DEBUG_VOLUMES
