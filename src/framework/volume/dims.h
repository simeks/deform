#pragma once

#include <stdint.h>

struct Dims
{
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

inline bool operator==(const Dims& l, const Dims& r)
{
    return (l.width == r.width && l.height == r.height && l.depth == r.depth);
}
inline bool operator!=(const Dims& l, const Dims& r)
{
    return !operator==(l, r);
}

