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

void init_default_settings(Settings& settings);

/*!
 * \brief Validate input and perform registration.
 *
 * This function validates the volumes and handles the registration
 * logic.
 *
 * #throws ValidationError
 */
stk::Volume registration(
        Settings& settings,
        std::vector<stk::Volume>& fixed_volumes,
        std::vector<stk::Volume>& moving_volumes,
        std::optional<stk::Volume> initial_deformation,
        std::optional<stk::Volume> constraint_mask,
        std::optional<stk::Volume> constraint_values,
        const int num_threads);
