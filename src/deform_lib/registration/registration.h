#pragma once

#include <deform_lib/registration/settings.h>

#include <stk/image/volume.h>

#include <vector>

/*!
 * \brief Ensure that the input volume properties (size, origin,
 *        spacing, direction) match the reference volume ones.
 *
 * @throws ValidationError
 */
void validate_volume_properties(
    const stk::Volume& vol,
    const stk::Volume& ref_vol,
    const std::string& name);

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
        const stk::Volume& fixed_mask,
        const stk::Volume& moving_mask,
        const std::vector<float3>& fixed_landmarks,
        const std::vector<float3>& moving_landmarks,
        const stk::Volume& initial_deformation,
        const stk::Volume& constraint_mask,
        const stk::Volume& constraint_values,
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        const stk::Volume& regularization_map,
#endif
        const int num_threads
#ifdef DF_USE_CUDA
        , bool use_gpu
#endif
);
