#include "hard_constraints.h"

#ifdef DF_ENABLE_HARD_CONSTRAINTS

#include <framework/math/float3.h>
#include <framework/math/int3.h>

VolumeUInt8 hard_constraints::downsample_mask_by_2(const VolumeUInt8& mask)
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
                    max = std::max(max, mask(src_p + subvoxels[i]));
                }
                result(x, y, z) = max; 
            }
        }
    }
    return result;
}
VolumeFloat3 hard_constraints::downsample_values_by_2(
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
                    if (mask(src_p + subvoxels[i]) > 0)
                    {
                        ++nmask;
                        val = val + values(src_p);
                    }
                }
                result(x, y, z) = 0.5f * val / float(nmask);
            }
        }
    }
    return result;
}

#endif // DF_ENABLE_HARD_CONSTRAINTS