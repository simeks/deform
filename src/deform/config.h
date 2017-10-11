#pragma once

#define DF_MAX_IMAGE_PAIR_COUNT 8
#define DF_ENABLE_VOXEL_CONSTRAINTS
//#define DF_OUTPUT_VOLUME_ENERGY

#ifdef DF_BUILD_DEBUG
// 0 : No debug
// 3 : 
// 4 : Spam
#define DF_DEBUG_LEVEL 2


#if DF_DEBUG_LEVEL >= 2
    #define DF_OUTPUT_VOLUME_ENERGY
#endif

#if DF_DEBUG_LEVEL >= 3
    #define DF_OUTPUT_DEBUG_VOLUMES

    // Checks whether the change in block energy corresponds to the change in total energy
    #define DF_ENABLE_BLOCK_ENERGY_CHECK 
#endif 

#if DF_DEBUG_LEVEL >= 4
    #define DF_DEBUG_BLOCK_CHANGE_COUNT
    #define DF_DEBUG_VOXEL_CHANGE_COUNT
#endif

#endif // DF_BUILD_DEBUG