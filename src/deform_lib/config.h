#pragma once

#define DF_MAX_IMAGE_PAIR_COUNT 8

//#define DF_ENABLE_REGULARIZATION_WEIGHT_MAP

// Use the SSE version of linear_at<float>
// Does not make any real difference in performance on msvc2017
// #define DF_ENABLE_SSE_LINEAR_AT

#ifdef _DEBUG
    #define DF_OUTPUT_VOLUME_ENERGY
    //#define DF_OUTPUT_DEBUG_VOLUMES
#endif
