#pragma once

// TODO: Cleanup

#include "types.h"

inline float3 operator+(const float3& l, const float3& r)
{
    return { l.x + r.x, l.y + r.y, l.z + r.z };
}
inline float3 operator-(const float3& l, const float3& r)
{
    return { l.x - r.x, l.y - r.y, l.z - r.z };
}

inline float3 operator*(float l, const float3& r)
{
    return { r.x * l, r.y * l, r.z * l };
}
