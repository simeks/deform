#pragma once

void initialize_regularization(
    stk::VolumeFloat3& def,
    const stk::VolumeUChar& constraints_mask,
    const stk::VolumeFloat3& constraints_values
    );
void do_regularization(
    stk::VolumeFloat3& def,
    const stk::VolumeUChar& constraints_mask,
    const stk::VolumeFloat3& constraints_values,
    float precision
    );
