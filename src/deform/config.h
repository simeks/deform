#pragma once

#ifdef DF_BUILD_DEBUG
// 0 : No debug
// 3 : 
// 4 : Spam
#define DF_DEBUG_LEVEL 3


#if DF_DEBUG_LEVEL >= 3
    #define DF_OUTPUT_DEBUG_VOLUMES
#endif 

#if DF_DEBUG_LEVEL >= 4
    #define DF_DEBUG_BLOCK_CHANGE_COUNT
    #define DF_DEBUG_VOXEL_CHANGE_COUNT
#endif

#endif // DF_BUILD_DEBUG