#pragma once

#include <deform_lib/registration/settings.h>

#include <stk/image/volume.h>

#include <optional>
#include <vector>

class ValidationError : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;
};

/*!
 * \brief Validate input and perform registration.
 *
 * This function validates the volumes and handles the registration
 * logic.
 *
 * #throws ValidationError
 */
stk::Volume registration(
        const Settings& settings,
        std::vector<stk::Volume>& fixed_volumes,
        std::vector<stk::Volume>& moving_volumes,
        const std::optional<stk::Volume> initial_deformation,
        const std::optional<stk::Volume> constraint_mask,
        const std::optional<stk::Volume> constraint_values,
        const int num_threads);
