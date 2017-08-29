#pragma once

#include "volume.h"

#include <assert.h>

template<typename T>
class VolumeHelper : public Volume
{
public:
    typedef T VoxelType;
    
    VolumeHelper(Volume& other);
    /// Creates a new volume of the specified size
    VolumeHelper(const Dims& size);
    /// Creates a new volume of the specified size and initializes it with the given value
    VolumeHelper(const Dims& size, const T& value);
    ~VolumeHelper();

    void fill(const T& value);

    VolumeHelper& operator=(VolumeHelper& other);
    VolumeHelper& operator=(Volume& other);

    const T& operator()(int x, int y, int z) const;
    T& operator()(int x, int y, int z);

    /// Offset in bytes to the specified element
    size_t offset(int x, int y, int z) const;

};


#include "volume_helper.inl"
