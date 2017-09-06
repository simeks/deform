#include "resample.h"

#include "debug/assert.h"
#include "filters/gaussian_filter.h"
#include "volume/volume_helper.h"

#include <algorithm>

namespace
{
    template<typename TVoxelType>
    VolumeHelper<TVoxelType> downsample_volume(const VolumeHelper<TVoxelType>& src, float scale)
    {
        assert(scale > 0.0f && scale < 1.0f);
        float inv_scale = 1.0f / scale;
        
        Dims old_dims = src.size();
        Dims new_dims
        {
            uint32_t(ceil(old_dims.width * scale)),
            uint32_t(ceil(old_dims.height * scale)),
            uint32_t(ceil(old_dims.depth * scale)),
        };

        VolumeHelper<TVoxelType> dest(new_dims);
        dest.set_origin(src.origin());
        
        float3 old_spacing = src.spacing();
        float3 new_spacing{
            old_spacing.x * inv_scale,
            old_spacing.y * inv_scale,
            old_spacing.z * inv_scale
        };
        dest.set_spacing(new_spacing);

    #pragma omp parallel for
        for (int z = 0; z < int(new_dims.depth); ++z)
        {
            for (int y = 0; y < int(new_dims.height); ++y)
            {
                for (int x = 0; x < int(new_dims.width); ++x)
                {
                    dest(x, y, z) = src(int(x*inv_scale), int(y*inv_scale), int(z*inv_scale));
                }
            }
        }
        return dest;
    }
}


Volume filters::downsample_volume_gaussian(const Volume& vol, float scale)
{
    assert(scale > 0.0f && scale < 1.0f);
    assert( vol.voxel_type() == voxel::Type_Float ||
            vol.voxel_type() == voxel::Type_Double);

    float3 spacing = vol.spacing();
    float unit_sigma = std::min(spacing.x, std::min(spacing.y, spacing.z));
    Volume filtered = filters::gaussian_filter_3d(vol, unit_sigma);

    switch (vol.voxel_type())
    {
    case voxel::Type_Float:
        return ::downsample_volume<float>(vol, scale);
    case voxel::Type_Double:
        return ::downsample_volume<double>(vol, scale);
    default:
        assert(false);
    };

    return Volume();
}

