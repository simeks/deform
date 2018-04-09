#pragma once

#include <smmintrin.h>

#include "log.h"

namespace sse
{
    inline void print(__m128 v)
    {
        alignas(16) float vf[4];
        _mm_store_ps(vf, v);
        LOG(Debug, "(%f, %f, %f, %f)\n", vf[0], vf[1], vf[2], vf[3]);
    }
}
