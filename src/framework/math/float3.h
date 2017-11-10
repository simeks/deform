#pragma once

// TODO: Cleanup

#include "types.h"

#include <math.h>

namespace math
{
    inline float length_squared(const float3& v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }
    inline float length(const float3& v)
    {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }
}

inline float3 operator+(const float3& l, const float3& r)
{
    return { l.x + r.x, l.y + r.y, l.z + r.z };
}
inline float3 operator-(const float3& l, const float3& r)
{
    return { l.x - r.x, l.y - r.y, l.z - r.z };
}

// element-wise multiplication
inline float3 operator*(const float3& l, const float3& r)
{
    return { l.x * r.x, l.y * r.y, l.z * r.z };
}
// element-wise division
inline float3 operator/(const float3& l, const float3& r)
{
    return { l.x / r.x, l.y / r.y, l.z / r.z };
}

inline float3 operator*(float l, const float3& r)
{
    return { r.x * l, r.y * l, r.z * l };
}
inline float3 operator/(const float3& l, float r)
{
    return { l.x / r, l.y / r, l.z / r };
}

