#include "volume_pyramid.h"
#include "voxel_constraints.h"

#include <framework/math/float3.h>
#include <framework/math/int3.h>

VolumeUInt8 voxel_constraints::downsample_mask_by_2(const VolumeUInt8& mask)
{
    /*
        Downsampling of mask volumes by a factor of 2. Designed specifically for binary masks.
        Each resulting downsampled voxel takes the max value from the corresponding subvoxels 
    */

    int3 subvoxels[] = {
        int3{0, 0, 0},
        int3{0, 0, 1},
        int3{0, 1, 0},
        int3{0, 1, 1},
        int3{1, 0, 0},
        int3{1, 0, 1},
        int3{1, 1, 0},
        int3{1, 1, 1}
    };

    Dims old_dims = mask.size();
    Dims new_dims{
        uint32_t(ceil(old_dims.width * 0.5f)),
        uint32_t(ceil(old_dims.height * 0.5f)),
        uint32_t(ceil(old_dims.depth * 0.5f)),
    };

    VolumeUInt8 result(new_dims);

    #pragma omp parallel for
    for (int z = 0; z < int(new_dims.depth); ++z)
    {
        for (int y = 0; y < int(new_dims.height); ++y)
        {
            for (int x = 0; x < int(new_dims.width); ++x)
            {
                int3 src_p{2*x, 2*y, 2*z};
                
                uint8_t max = 0;
                for (int i = 0; i < 8; ++i)
                {
					int3 p = 2 * src_p + subvoxels[i];
					if (p.x >= int(old_dims.width) ||
						p.y >= int(old_dims.height) ||
						p.z >= int(old_dims.depth))
						continue;

                    max = std::max(max, mask(src_p + subvoxels[i]));
                }
                result(x, y, z) = max; 
            }
        }
    }
    return result;
}
VolumeFloat3 voxel_constraints::downsample_values_by_2(
    const VolumeUInt8& mask, const VolumeFloat3& values)
{
    /*
        Downsamples a constraint vector field.
        The value of each downsampled voxel is calculated as the mean of all subvoxels
            that are flagged as constraints (1s in the mask). 
    */
    
    assert(mask.size() == values.size());

    int3 subvoxels[] = {
        int3{0, 0, 0},
        int3{0, 0, 1},
        int3{0, 1, 0},
        int3{0, 1, 1},
        int3{1, 0, 0},
        int3{1, 0, 1},
        int3{1, 1, 0},
        int3{1, 1, 1}
    };
    
    Dims old_dims = mask.size();
    Dims new_dims{
        uint32_t(ceil(old_dims.width * 0.5f)),
        uint32_t(ceil(old_dims.height * 0.5f)),
        uint32_t(ceil(old_dims.depth * 0.5f)),
    };

    VolumeFloat3 result(new_dims, float3{0});

    #pragma omp parallel for
    for (int z = 0; z < int(new_dims.depth); ++z)
    {
        for (int y = 0; y < int(new_dims.height); ++y)
        {
            for (int x = 0; x < int(new_dims.width); ++x)
            {
                int3 src_p{2*x, 2*y, 2*z};

                int nmask = 0;
                float3 val{0};
                for (int i = 0; i < 8; ++i)
                {
					int3 p = src_p + subvoxels[i];
					if (p.x >= int(old_dims.width) ||
						p.y >= int(old_dims.height) ||
						p.z >= int(old_dims.depth))
						continue;

                    if (mask(p) > 0)
                    {
                        ++nmask;
                        val = val + values(p);
                    }
                }
                result(x, y, z) = 0.5f * val / float(nmask);
            }
        }
    }
    return result;
}


void voxel_constraints::build_pyramids(const VolumeUInt8& mask, const VolumeFloat3& values,
    int num_levels, VolumePyramid& mask_pyramid, VolumePyramid& values_pyramid)
{
    mask_pyramid.set_level_count(num_levels);
    values_pyramid.set_level_count(num_levels);
    
    mask_pyramid.set_volume(0, mask);
    values_pyramid.set_volume(0, values);

    for (int i = 0; i < num_levels-1; ++i)
    {
        VolumeUInt8 prev_mask = mask_pyramid.volume(i);
        VolumeFloat3 prev_values = values_pyramid.volume(i);

        mask_pyramid.set_volume(i+1, downsample_mask_by_2(prev_mask));
        values_pyramid.set_volume(i+1, downsample_values_by_2(prev_mask, prev_values));
    }
}
