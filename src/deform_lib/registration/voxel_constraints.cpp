#include "volume_pyramid.h"
#include "voxel_constraints.h"

#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>

stk::VolumeUChar voxel_constraints::downsample_mask_by_2(const stk::VolumeUChar& mask)
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

    dim3 old_dims = mask.size();
    dim3 new_dims{
        uint32_t(ceil(old_dims.x * 0.5f)),
        uint32_t(ceil(old_dims.y * 0.5f)),
        uint32_t(ceil(old_dims.z * 0.5f)),
    };

    stk::VolumeUChar result(new_dims);

    #pragma omp parallel for
    for (int z = 0; z < int(new_dims.z); ++z) {
        for (int y = 0; y < int(new_dims.y); ++y) {
            for (int x = 0; x < int(new_dims.x); ++x) {
                int3 src_p{2*x, 2*y, 2*z};
                
                uint8_t max = 0;
                for (int i = 0; i < 8; ++i) {
                    int3 p = 2 * src_p + subvoxels[i];
                    if (p.x >= int(old_dims.x) ||
                        p.y >= int(old_dims.y) ||
                        p.z >= int(old_dims.z))
                        continue;

                    max = std::max(max, mask(src_p + subvoxels[i]));
                }
                result(x, y, z) = max; 
            }
        }
    }
    return result;
}
stk::VolumeFloat3 voxel_constraints::downsample_values_by_2(
    const stk::VolumeUChar& mask, const stk::VolumeFloat3& values)
{
    /*
        Downsamples a constraint vector field.
        The value of each downsampled voxel is calculated as the mean of all subvoxels
            that are flagged as constraints (1s in the mask). 
    */
    
    ASSERT(mask.size() == values.size());

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
    
    dim3 old_dims = mask.size();
    dim3 new_dims{
        uint32_t(ceil(old_dims.x * 0.5f)),
        uint32_t(ceil(old_dims.y * 0.5f)),
        uint32_t(ceil(old_dims.z * 0.5f)),
    };

    stk::VolumeFloat3 result(new_dims, float3{0});

    #pragma omp parallel for
    for (int z = 0; z < int(new_dims.z); ++z) {
        for (int y = 0; y < int(new_dims.y); ++y) {
            for (int x = 0; x < int(new_dims.x); ++x) {
                int3 src_p{2*x, 2*y, 2*z};

                int nmask = 0;
                float3 val{0};
                for (int i = 0; i < 8; ++i) {
                    int3 p = src_p + subvoxels[i];
                    if (p.x >= int(old_dims.x) ||
                        p.y >= int(old_dims.y) ||
                        p.z >= int(old_dims.z))
                        continue;

                    if (mask(p) > 0) {
                        ++nmask;
                        val = val + values(p);
                    }
                }
                // TODO: Div by zero?
                result(x, y, z) = 0.5f * val / float(nmask);
            }
        }
    }
    return result;
}


void voxel_constraints::build_pyramids(
    const stk::VolumeUChar& mask, 
    const stk::VolumeFloat3& values,
    int num_levels, 
    VolumePyramid& mask_pyramid, 
    VolumePyramid& values_pyramid)
{
    mask_pyramid.set_level_count(num_levels);
    values_pyramid.set_level_count(num_levels);
    
    mask_pyramid.set_volume(0, mask);
    values_pyramid.set_volume(0, values);

    for (int i = 0; i < num_levels-1; ++i) {
        stk::VolumeUChar prev_mask = mask_pyramid.volume(i);
        stk::VolumeFloat3 prev_values = values_pyramid.volume(i);

        mask_pyramid.set_volume(i+1, downsample_mask_by_2(prev_mask));
        values_pyramid.set_volume(i+1, downsample_values_by_2(prev_mask, prev_values));
    }
}
