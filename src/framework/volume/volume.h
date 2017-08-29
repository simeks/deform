#pragma once

#include "dims.h"
#include "voxel.h"

#include <memory>
#include <vector>

#ifdef DF_ENABLE_CUDA
struct GpuVolume;
#endif // DF_ENABLE_CUDA

struct VolumeData
{
    VolumeData();
    VolumeData(size_t size);
    ~VolumeData();

    uint8_t* data;
    size_t size;
};

class Volume
{
public:
    Volume();
    Volume(const Dims& size, uint8_t voxel_type, uint8_t* data = nullptr);
    ~Volume();

    /// Release any allocated data the volume is holding
    /// This makes the volume object invalid
    void release();

    /// Clones this volume
    Volume clone() const;

    /// Attempts to convert this volume to the specified format,
    ///     If this volume already is of the specified format it will just return itself.
    ///     If not a copied converted version will be returned.
    /// Note:
    ///     You can only convert between volumes with the same number of components per voxel.
    ///     I.e. you cannot convert from Float3 to Int2.
    Volume as_type(uint8_t type) const;

    /// Returns true if the volume is allocated and ready for use
    bool valid() const;

    /// Raw pointer to the volume data
    void* ptr();

    /// Raw pointer to the volume data
    void const* ptr() const;

    uint8_t voxel_type() const;
    const Dims& size() const;

    /// @remark This does not copy the data, use clone if you want a separate copy.
    Volume(const Volume& other);
    Volume& operator=(const Volume& other);

#ifdef DF_ENABLE_CUDA
    /// Creates a new volume on the CPU side and downloads the given volume from the gpu into it.
    Volume(const GpuVolume& gpu_volume);

    /// Uploads this volume to a newly allocated GPU volume
    /// @remark Requires both volumes to be of same size and type
    /// @return Handle to newly created GPU volume
    GpuVolume upload() const;

    /// Uploads this volume to given GPU volume
    /// @remark Requires both volumes to be of same size and type
    void upload(const GpuVolume& gpu_volume) const;

    /// Downloads the given volume into this volume
    /// @remark Requires both volumes to be of same size and type
    void download(const GpuVolume& gpu_volume);
#endif // DF_ENABLE_CUDA


protected:
    void allocate(const Dims& size, uint8_t voxel_type);

    std::shared_ptr<VolumeData> _data;
    void* _ptr; // Pointer to a location in _data
    size_t _stride; // Size of a single row in bytes

    Dims _size;
    uint8_t _voxel_type;
};
