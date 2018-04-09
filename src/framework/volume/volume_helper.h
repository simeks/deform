#pragma once

#include "volume.h"

#include <algorithm>
#include <assert.h>
#include <float.h>
#include <math.h>

#ifdef DF_ENABLE_SSE_LINEAR_AT
    #include <smmintrin.h>
#endif


namespace volume
{
    enum BorderMode
    {
        Border_Constant, // Zero padding outside volume
        Border_Replicate
    };
}

template<typename T>
class VolumeHelper : public Volume
{
public:
    typedef T TVoxelType;
    
    /// Creates a null (invalid) volume
    VolumeHelper();
    VolumeHelper(const Volume& other);
    /// Creates a new volume of the specified size
    VolumeHelper(const Dims& size);
    /// Creates a new volume of the specified size and initializes it with the given value
    VolumeHelper(const Dims& size, const T& value);
    ~VolumeHelper();

    void fill(const T& value);

    T at(int x, int y, int z, volume::BorderMode border_mode) const;
    T at(int3 p, volume::BorderMode border_mode) const;

    T linear_at(float x, float y, float z, volume::BorderMode border_mode) const;
    T linear_at(float3 p, volume::BorderMode border_mode) const;

    VolumeHelper& operator=(const VolumeHelper& other);
    VolumeHelper& operator=(const Volume& other);

    const T& operator()(int x, int y, int z) const;
    T& operator()(int x, int y, int z);
    
    const T& operator()(const int3& p) const;
    T& operator()(const int3& p);

    /// Offset in bytes to the specified element
    size_t offset(int x, int y, int z) const;

    /// Finds min and max in volume
    /// min [out]
    /// max [out]
    void min_max(T& min, T& max) const;

};

typedef VolumeHelper<uint8_t> VolumeUInt8;

typedef VolumeHelper<float> VolumeFloat;
typedef VolumeHelper<double> VolumeDouble;
typedef VolumeHelper<float3> VolumeFloat3;

#include "volume_helper.inl"
