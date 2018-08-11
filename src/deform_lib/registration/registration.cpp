#include <deform_lib/arg_parser.h>
#include <deform_lib/config.h>
#include <deform_lib/defer.h>
#include <deform_lib/filters/resample.h>
#include <deform_lib/jacobian.h>
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
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <optional>
#include <string>
#include <vector>

#include "registration.h"

/// name : Name for printout
static void validate_volume_properties(
    const stk::Volume& vol,
    const dim3& expected_dims,
    const float3& expected_origin,
    const float3& expected_spacing,
    const std::string& name)
{
    dim3 dims = vol.size();
    float3 origin = vol.origin();
    float3 spacing = vol.spacing();
    std::ostringstream osstr;

    if (dims != expected_dims)
    {
        osstr << "Dimension mismatch for " << name
              << " (size: " << dims << ", expected: "
              << expected_dims << ")";
        throw ValidationError(osstr.str());
    }

    // arbitrary epsilon but should suffice
    if (fabs(origin.x - expected_origin.x) > 0.0001f ||
        fabs(origin.y - expected_origin.y) > 0.0001f ||
        fabs(origin.z - expected_origin.z) > 0.0001f)
    {
        osstr << "Origin mismatch for " << name
              << " (origin: " << origin << ", expected: "
              << expected_origin << ")";
        throw ValidationError(osstr.str());
    }

    if (fabs(spacing.x - expected_spacing.x) > 0.0001f ||
        fabs(spacing.y - expected_spacing.y) > 0.0001f ||
        fabs(spacing.z - expected_spacing.z) > 0.0001f)
    {
        osstr << "Spacing mismatch for " << name
              << " (spacing: " << spacing << ", expected: "
              << expected_spacing << ")";
        throw ValidationError(osstr.str());
    }
}

stk::Volume registration(
        const Settings& settings,
        std::vector<stk::Volume>& fixed_volumes,
        std::vector<stk::Volume>& moving_volumes,
        const std::optional<std::vector<float3>> fixed_landmarks,
        const std::optional<std::vector<float3>> moving_landmarks,
        const std::optional<stk::Volume> initial_deformation,
        const std::optional<stk::Volume> constraint_mask,
        const std::optional<stk::Volume> constraint_values,
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
        if (!fixed.valid()) {
            throw ValidationError("Invalid fixed volume at index " + std::to_string(i));
        }
        stk::Volume& moving = moving_volumes[i];
        if (!moving.valid()) {
            throw ValidationError("Invalid moving volume at index " + std::to_string(i));
        }

        if (fixed.voxel_type() != moving.voxel_type()) {
            throw ValidationError("Mismatch in voxel type between pairs at index "
                                  + std::to_string(i) + ", "
                                  + "fixed type '" + stk::as_string(fixed.voxel_type()) + "', "
                                  + "moving type '" + stk::as_string(moving.voxel_type()) + "'.");
        }

        if (!fixed_ref.valid() || !moving_ref.valid()) {
            fixed_ref = fixed;
            moving_ref = moving;
        }
        else {
            validate_volume_properties(fixed, fixed_ref.size(),
                    fixed_ref.origin(), fixed_ref.spacing(), fixed_id);
            validate_volume_properties(moving, moving_ref.size(),
                    moving_ref.origin(), moving_ref.spacing(), moving_id);
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
                throw ValidationError("Normalize only supported on volumes of type float or double");
            }
        }

        // It's the only available fn for now
        auto downsample_fn = filters::downsample_volume_by_2;

        engine.set_image_pair(static_cast<int>(i), fixed, moving, downsample_fn);
    }

    if (initial_deformation.has_value()) {
        if (!initial_deformation.value().valid()) {
            throw ValidationError("Invalid initial deformation volume");
        }

        validate_volume_properties(initial_deformation.value(), fixed_ref.size(),
                fixed_ref.origin(), fixed_ref.spacing(), "initial deformation field");

        engine.set_initial_deformation(initial_deformation.value());
    }

    if (constraint_mask.has_value() && constraint_values.has_value()) {
        if (!constraint_mask.value().valid()) {
            throw ValidationError("Invalid constraint mask volume");
        }

        if (!constraint_values.value().valid()) {
            throw ValidationError("Invalid constraint values volume");
        }

        validate_volume_properties(constraint_mask.value(), fixed_ref.size(),
                fixed_ref.origin(), fixed_ref.spacing(), "constraint mask");

        validate_volume_properties(constraint_values.value(), fixed_ref.size(),
                fixed_ref.origin(), fixed_ref.spacing(), "constraint values");

        engine.set_voxel_constraints(constraint_mask.value(), constraint_values.value());
    }

    if (fixed_landmarks.has_value() || moving_landmarks.has_value()) {
        if (!fixed_landmarks.has_value() || !moving_landmarks.has_value()) {
            throw ValidationError("Landmarks must be specified for both fixed and moving");
        }

        if (fixed_landmarks.value().size() != moving_landmarks.value().size()) {
            throw ValidationError("The number of fixed and moving landmarks must match");
        }

        engine.set_landmarks(fixed_landmarks.value(), moving_landmarks.value());
    }

    using namespace std::chrono;
    auto t_start = high_resolution_clock::now();
    stk::Volume def = engine.execute();
    auto t_end = high_resolution_clock::now();
    int elapsed = int(round(duration_cast<duration<double>>(t_end - t_start).count()));
    LOG(Info) << "Registration completed in " << elapsed / 60 << ":" << std::right << std::setw(2) << std::setfill('0') << elapsed % 60;

    return def;
}
