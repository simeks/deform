#pragma once

#include <stk/image/volume.h>

#include <vector>

struct LevelContext
{
    int level;

    stk::VolumeFloat3 initial_displacement;

    std::vector<stk::Volume> fixed_volumes;
    std::vector<stk::Volume> moving_volumes;
    
    stk::VolumeUChar constraint_mask;
    stk::VolumeFloat3 constraint_values;
    
    float regularization_weight;

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    stk::VolumeFloat& regularization_weight_map;
#endif
};
