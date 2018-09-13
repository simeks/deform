#pragma once

#include <deform_lib/registration/settings.h>

#include <stk/image/volume.h>

#include <optional>
#include <vector>

/*!
 * \brief Validate input and perform registration.
 *
 * This function validates the volumes and handles the registration
 * logic.
 *
 * @throws ValidationError
 */
stk::Volume registration(
        const Settings& settings,
        std::vector<stk::Volume>& fixed_volumes,
        std::vector<stk::Volume>& moving_volumes,
        const std::optional<std::vector<float3>> fixed_landmarks,
        const std::optional<std::vector<float3>> moving_landmarks,
        const std::optional<stk::Volume> initial_deformation,
        const std::optional<stk::Volume> constraint_mask,
        const std::optional<stk::Volume> constraint_values,
        const int num_threads
#ifdef DF_USE_CUDA
        , bool use_gpu
#endif
);
