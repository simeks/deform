#pragma once

inline int3 operator*(const int l, const int3& r)
{
    return int3{l*r.x, l*r.y, l*r.z};
}
/// Element-wise multiplication
inline int3 operator*(const int3& l, const int3& r)
{
    return int3{l.x*r.x, l.y*r.y, l.z*r.z};
}
inline int3 operator+(const int3& l, const int3& r)
{
    return int3{l.x+r.x, l.y+r.y, l.z+r.z};
}
inline int3 operator-(const int3& l, const int3& r)
{
    return int3{l.x-r.x, l.y-r.y, l.z-r.z};
}