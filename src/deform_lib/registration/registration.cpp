#include <deform_lib/arg_parser.h>
#include <deform_lib/config.h>
#include <deform_lib/defer.h>
#include <deform_lib/filters/resample.h>
#include <deform_lib/jacobian.h>
#include <deform_lib/platform/file_path.h>
#include <deform_lib/platform/timer.h>
#include <deform_lib/registration/registration_engine.h>
#include <deform_lib/registration/settings.h>
#include <deform_lib/registration/transform.h>
#include <deform_lib/registration/volume_pyramid.h>

#include <stk/common/assert.h>
#include <stk/common/log.h>
#include <stk/filters/normalize.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <optional>
#include <string>
#include <vector>

#include "registration.h"

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

// Creates some default settings settings SSD as metric for all image pairs
void init_default_settings(Settings& settings)
{
    // Assume that Settings-constructor has initialized some decent defaults,
    //  we only need to define the matching metrics

    for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
        settings.image_slots[i].cost_function = Settings::ImageSlot::CostFunction_SSD;
        settings.image_slots[i].resample_method = Settings::ImageSlot::Resample_Gaussian;
        settings.image_slots[i].normalize = true;
    }
}

stk::Volume registration(
        Settings& settings,
        std::vector<stk::Volume>& fixed_volumes,
        std::vector<stk::Volume>& moving_volumes,
        std::optional<stk::Volume> initial_deformation,
        std::optional<stk::Volume> constraint_mask,
        std::optional<stk::Volume> constraint_values,
        const int num_threads = 0
        )
{
    LOG(Info) << "Running registration";

    if (num_threads > 0) {
        LOG(Info) << "Number of threads: " << num_threads;
        omp_set_num_threads(num_threads);
    }

    print_registration_settings(settings);
    RegistrationEngine engine(settings);

    // Rules:
    // * All volumes for the same subject (i.e. fixed or moving) must have the
    //      same dimensions.
    // * All volumes for the same subject (i.e. fixed or moving) need to have
    //      the same origin and spacing.
    // * For simplicity any given initial deformation field must match the
    //      fixed image properties (size, origin, spacing).
    // * Pairs must have a matching data type
    // If hard constraints are enabled:
    // * Constraint mask and values must match fixed image

    stk::Volume fixed_ref; // Reference volume for validation
    stk::Volume moving_ref; // Reference volume for computing the jacobian

    for (size_t i = 0; i < fixed_volumes.size() && i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
        std::string fixed_id = "fixed" + std::to_string(i);
        std::string moving_id = "moving" + std::to_string(i);

        stk::Volume& fixed = fixed_volumes[i];
        if (!fixed.valid())
            throw std::runtime_error("Invalid fixed volume"); // FIXME
        stk::Volume& moving = moving_volumes[i];
        if (!moving.valid())
            throw std::runtime_error("Invalid moving volume"); // FIXME

        if (fixed.voxel_type() != moving.voxel_type()) {
            LOG(Error) << "Mismatch in voxel type between pairs at index " << i;
            throw std::runtime_error("Mismatching voxel type"); // FIXME
        }

        if (!fixed_ref.valid() || !moving_ref.valid()) {
            fixed_ref = fixed;
            moving_ref = moving;
        }
        else {
            if (!validate_volume_properties(fixed, fixed_ref.size(),
                    fixed_ref.origin(), fixed_ref.spacing(), fixed_id)) {
                throw std::runtime_error("Invalid fixed volume properties"); // FIXME
            }
            if (!validate_volume_properties(moving, moving_ref.size(),
                    moving_ref.origin(), moving_ref.spacing(), moving_id)) {
                throw std::runtime_error("Invalid moving volume properties"); // FIXME
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
                throw std::runtime_error("Normalisation unsupported"); // FIXME
            }
        }

        // It's the only available fn for now
        auto downsample_fn = filters::downsample_volume_gaussian;

        engine.set_image_pair(i, fixed, moving, downsample_fn);
    }

    if (initial_deformation.has_value()) {
        if (!initial_deformation.value().valid())
            throw std::runtime_error("Invalid initial deformation volume"); // FIXME

        if (!validate_volume_properties(initial_deformation.value(), fixed_ref.size(),
                fixed_ref.origin(), fixed_ref.spacing(), "initial deformation field"))
            throw std::runtime_error("Invalid initial deformation properties"); // FIXME

        engine.set_initial_deformation(initial_deformation.value());
    }

    if (constraint_mask.has_value() && constraint_values.has_value()) {
        if (!constraint_mask.value().valid())
            throw std::runtime_error("Invalid constraint mask volume"); // FIXME

        if (!constraint_values.value().valid())
            throw std::runtime_error("Invalid constraint values volume"); // FIXME

        if (!validate_volume_properties(constraint_mask.value(), fixed_ref.size(),
                fixed_ref.origin(), fixed_ref.spacing(), "constraint mask"))
            throw std::runtime_error("Invalid constraint mask properties"); // FIXME

        if (!validate_volume_properties(constraint_values.value(), fixed_ref.size(),
                fixed_ref.origin(), fixed_ref.spacing(), "constraint values"))
            throw std::runtime_error("Invalid constraint values properties"); // FIXME

        engine.set_voxel_constraints(constraint_mask.value(), constraint_values.value());
    }

    double t_start = timer::seconds();
    stk::Volume def = engine.execute();
    double t_end = timer::seconds();
    int elapsed = int(round(t_end - t_start));
    LOG(Info) << "Registration completed in " << elapsed / 60 << ":" << std::setw(2) << std::setfill('0') << elapsed % 60;

    return def;
}
