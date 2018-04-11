#pragma once

#include <stdint.h>

#include <framework/math/int3.h>

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

// Check whether the given point is inside the given range
inline bool is_inside(const Dims& dims, const int3& p)
{
    return (p.x >= 0 && p.x < int(dims.width) && p.y >= 0 && p.y < int(dims.height) && p.z >= 0 && p.z < int(dims.depth));
}

