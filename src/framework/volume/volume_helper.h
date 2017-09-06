#pragma once

#include "volume.h"

#include <assert.h>

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
    
    VolumeHelper(const Volume& other);
    /// Creates a new volume of the specified size
    VolumeHelper(const Dims& size);
    /// Creates a new volume of the specified size and initializes it with the given value
    VolumeHelper(const Dims& size, const T& value);
    ~VolumeHelper();

    void fill(const T& value);

    T at(int x, int y, int z, volume::BorderMode border_mode) const;
    T linear_at(float x, float y, float z, volume::BorderMode border_mode) const;

    VolumeHelper& operator=(VolumeHelper& other);
    VolumeHelper& operator=(Volume& other);

    const T& operator()(int x, int y, int z) const;
    T& operator()(int x, int y, int z);

    /// Offset in bytes to the specified element
    size_t offset(int x, int y, int z) const;

};


#include "volume_helper.inl"
