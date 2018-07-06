#include <deform_lib/registration/settings.h>

#include <stk/image/volume.h>

#include <optional>
#include <vector>

bool validate_volume_properties(
        const stk::Volume& vol,
        const dim3& expected_dims,
        const float3& expected_origin,
        const float3& expected_spacing,
        const std::string& name);

void init_default_settings(Settings& settings);

stk::Volume registration(
        Settings& settings,
        std::vector<stk::Volume>& fixed_volumes,
        std::vector<stk::Volume>& moving_volumes,
        std::optional<stk::Volume> initial_deformation,
        std::optional<stk::Volume> constraint_mask,
        std::optional<stk::Volume> constraint_values,
        const int num_threads);
