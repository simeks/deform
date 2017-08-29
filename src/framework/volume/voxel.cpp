#include "voxel.h"

#include <assert.h>

namespace voxel
{
    size_t size(uint8_t type)
    {
        switch (type)
        {
        case Type_Float:
            return sizeof(float);
        case Type_Float2:
            return sizeof(float) * 2;
        case Type_Float3:
            return sizeof(float) * 3;
        case Type_Float4:
            return sizeof(float) * 4;
        case Type_Double:
            return sizeof(double);
        case Type_Double2:
            return sizeof(double) * 2;
        case Type_Double3:
            return sizeof(double) * 3;
        case Type_Double4:
            return sizeof(double) * 4;
        case Type_UChar:
            return sizeof(uint8_t);
        case Type_UChar2:
            return sizeof(uint8_t) * 2;
        case Type_UChar3:
            return sizeof(uint8_t) * 3;
        case Type_UChar4:
            return sizeof(uint8_t) * 4;
        default:
            assert(false);
        };
        return 0;
    }
    int num_components(uint8_t type)
    {
        switch (type)
        {
        case Type_Float:
        case Type_Double:
        case Type_UChar:
            return 1;
        case Type_Float2:
        case Type_Double2:
        case Type_UChar2:
            return 2;
        case Type_Float3:
        case Type_Double3:
        case Type_UChar3:
            return 3;
        case Type_Float4:
        case Type_Double4:
        case Type_UChar4:
            return 4;
        default:
            assert(false);
        };
        return 0;
    }
    uint8_t base_type(uint8_t type)
    {
        switch (type)
        {
        case Type_Float:
        case Type_Float2:
        case Type_Float3:
        case Type_Float4:
            return Type_Float;
        case Type_Double:
        case Type_Double2:
        case Type_Double3:
        case Type_Double4:
            return Type_Double;
        case Type_UChar:
        case Type_UChar2:
        case Type_UChar3:
        case Type_UChar4:
            return Type_UChar;
        default:
            assert(false);
        };
        return Type_Unknown;
    }
}
