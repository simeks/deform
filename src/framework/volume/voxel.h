#pragma once

#include <stdint.h>
#include "types.h"

namespace voxel
{
    enum Type : uint8_t
    {
        Type_Unknown = 0,
        Type_Float,
        Type_Float2,
        Type_Float3,
        Type_Float4,
        Type_Double,
        Type_Double2,
        Type_Double3,
        Type_Double4,
        Type_UChar,
        Type_UChar2,
        Type_UChar3,
        Type_UChar4
    };
    
    /// Returns the total size in bytes of the specified type
    size_t size(uint8_t type);
    
    /// Returns the number of components of the specified type
    int num_components(uint8_t type);

    /// Returns the base type of a type, i.e. Float3 -> Float
    uint8_t base_type(uint8_t type);
}
template<typename T>
struct voxel_type
{
    typedef T Type;
    enum {
        type_id = voxel::Type_Unknown
    };
};

#define VOXEL_TYPE_TRAIT(T, Id) \
    template<> \
    struct voxel_type<T> \
    { \
        typedef T Type; \
        enum { \
            type_id = Id \
        }; \
    };

VOXEL_TYPE_TRAIT(float, voxel::Type_Float);
VOXEL_TYPE_TRAIT(float2, voxel::Type_Float2);
VOXEL_TYPE_TRAIT(float3, voxel::Type_Float3);
VOXEL_TYPE_TRAIT(float4, voxel::Type_Float4);

VOXEL_TYPE_TRAIT(double, voxel::Type_Double);
VOXEL_TYPE_TRAIT(double2, voxel::Type_Double2);
VOXEL_TYPE_TRAIT(double3, voxel::Type_Double3);
VOXEL_TYPE_TRAIT(double4, voxel::Type_Double4);

VOXEL_TYPE_TRAIT(uint8_t, voxel::Type_UChar);
VOXEL_TYPE_TRAIT(uchar2, voxel::Type_UChar2);
VOXEL_TYPE_TRAIT(uchar3, voxel::Type_UChar3);
VOXEL_TYPE_TRAIT(uchar4, voxel::Type_UChar4);

#undef VOXEL_TYPE_TRAIT