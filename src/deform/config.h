#pragma once


// 0 : No debug
// 3 : 
// 4 : Spam
#define DF_DEBUG_LEVEL 3


#if DF_DEBUG_LEVEL >= 3
    #define DF_OUTPUT_DEBUG_VOLUMES

#endif // DF_DEBUG_LEVEL >= 3

