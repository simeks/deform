#ifdef DF_ENABLE_CUDA

#include <cuda_runtime.h>

#include "gpu_volume.h"
#include "helper_cuda.h"
#include "volume.h"
#include "voxel.h"

#include <assert.h>

namespace
{
    cudaChannelFormatDesc create_format_desc(uint8_t voxel_type)
    {
        switch (voxel_type)
        {
        case voxel::Type_Float:
            return cudaCreateChannelDesc<float>();
        case voxel::Type_Float2:
            return cudaCreateChannelDesc<float2>();
        case voxel::Type_Float3:
            return cudaCreateChannelDesc<float3>();
        case voxel::Type_Float4:
            return cudaCreateChannelDesc<float4>();
        case voxel::Type_UChar:
            return cudaCreateChannelDesc<uchar1>();
        case voxel::Type_UChar2:
            return cudaCreateChannelDesc<uchar2>();
        case voxel::Type_UChar3:
            return cudaCreateChannelDesc<uchar3>();
        case voxel::Type_UChar4:
            return cudaCreateChannelDesc<uchar4>();
        default:
            assert(false);
        };
        return{ 0 };
    }
}

namespace gpu
{
    GpuVolume allocate_volume(uint8_t voxel_type, const Dims& size, uint32_t flags)
    {
        GpuVolume vol = { 0 };
        vol.size = { size.width, size.height, size.depth };
        vol.format_desc = create_format_desc(voxel_type);

        checkCudaErrors(cudaMalloc3DArray(&vol.ptr, &vol.format_desc, 
            { size.width, size.height, size.depth }, flags));

        return vol;
    }
    void release_volume(GpuVolume& vol)
    {
        if (vol.ptr == NULL) // not allocated
            return;

        checkCudaErrors(cudaFreeArray(vol.ptr));
        vol.ptr = NULL;
        vol.size = { 0, 0, 0 };
    }
    uint8_t voxel_type(const GpuVolume& vol)
    {
        int num_comp = 0;
        if (vol.format_desc.x > 0) ++num_comp;
        if (vol.format_desc.y > 0) ++num_comp;
        if (vol.format_desc.z > 0) ++num_comp;
        if (vol.format_desc.w > 0) ++num_comp;
        
        if (vol.format_desc.f == cudaChannelFormatKindFloat)
        {
            if (vol.format_desc.x != 32)
                assert(false && "Unsupported format");

            uint8_t voxel_type = voxel::Type_Unknown;
            if (num_comp == 1) voxel_type = voxel::Type_Float;
            if (num_comp == 2) voxel_type = voxel::Type_Float2;
            if (num_comp == 3) voxel_type = voxel::Type_Float3;
            if (num_comp == 4) voxel_type = voxel::Type_Float4;

            return voxel_type;
        }
        else if (vol.format_desc.f == cudaChannelFormatKindUnsigned)
        {
            if (vol.format_desc.x == 8)
            {
                uint8_t voxel_type = voxel::Type_Unknown;
                if (num_comp == 1) voxel_type = voxel::Type_UChar;
                if (num_comp == 2) voxel_type = voxel::Type_UChar2;
                if (num_comp == 3) voxel_type = voxel::Type_UChar3;
                if (num_comp == 4) voxel_type = voxel::Type_UChar4;
                return voxel_type;
            }
        }

        assert(false && "Unsupported format");
        return voxel::Type_Unknown;
    }
}
#endif // DF_ENABLE_CUDA