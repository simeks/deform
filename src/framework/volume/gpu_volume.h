#pragma once

#ifdef DF_ENABLE_CUDA

#include "dims.h"

#include <channel_descriptor.h>
#include <stdint.h>

struct GpuVolume
{
    struct cudaArray* ptr;
    Dims size;
    cudaChannelFormatDesc format_desc;
};

namespace gpu
{
    enum Flags
    {
        Flag_BindAsSurface = cudaArraySurfaceLoadStore
    };

    /// Allocates a GPU volume of the specified type and size
    /// @param voxel_type : See Volume::VoxelType
    GpuVolume allocate_volume(uint8_t voxel_type, const Dims& size, 
        uint32_t flags = 0);

    /// Releases a volume allocated with allocate_volume
    void release_volume(GpuVolume& vol);

    /// Returns Volume::VoxelType matching the specified GpuVolume
    uint8_t voxel_type(const GpuVolume& vol);
}
#endif // DF_ENABLE_CUDA