#pragma once

stk::VolumeFloat3 regularization(
    const stk::VolumeFloat3& df,
    float precision,
    int pyramid_levels,
    stk::VolumeUChar constraints_mask,
    stk::VolumeFloat3 constraints_values
);

