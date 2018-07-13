#include "resample.h"

#include "gaussian_filter.h"

#include <stk/common/assert.h>
#include <stk/image/volume.h>
#include <stk/math/float3.h>

#include <algorithm>

namespace
{
    template<typename TVoxelType>
    stk::VolumeHelper<TVoxelType> downsample_volume(const stk::VolumeHelper<TVoxelType>& src, float scale)
    {
        ASSERT(scale > 0.0f && scale < 1.0f);
        float inv_scale = 1.0f / scale;
        
        dim3 old_dims = src.size();
        dim3 new_dims {
            uint32_t(ceil(old_dims.x * scale)),
            uint32_t(ceil(old_dims.y * scale)),
            uint32_t(ceil(old_dims.z * scale)),
        };

        stk::VolumeHelper<TVoxelType> dest(new_dims);
        dest.set_origin(src.origin());
        
        float3 old_spacing = src.spacing();
        float3 new_spacing {
            old_spacing.x * (old_dims.x / float(new_dims.x)),
            old_spacing.y * (old_dims.y / float(new_dims.y)),
            old_spacing.z * (old_dims.z / float(new_dims.z))
        };
        dest.set_spacing(new_spacing);

        #pragma omp parallel for
        for (int z = 0; z < int(new_dims.z); ++z) {
            for (int y = 0; y < int(new_dims.y); ++y) {
                for (int x = 0; x < int(new_dims.x); ++x) {
                    dest(x, y, z) = src(int(x*inv_scale), int(y*inv_scale), int(z*inv_scale));
                }
            }
        }
        return dest;
    }
}


stk::Volume filters::downsample_volume_gaussian(const stk::Volume& vol, float scale)
{
    ASSERT(scale > 0.0f && scale < 1.0f);
    FATAL_IF(vol.voxel_type() != stk::Type_Float &&
             vol.voxel_type() != stk::Type_Double)
        << "Unsupported voxel format";

    float3 spacing = vol.spacing();
    float unit_sigma = std::min(spacing.x, std::min(spacing.y, spacing.z));
    stk::Volume filtered = filters::gaussian_filter_3d(vol, unit_sigma);

    switch (vol.voxel_type()) {
    case stk::Type_Float:
        return ::downsample_volume<float>(filtered, scale);
    case stk::Type_Double:
        return ::downsample_volume<double>(filtered, scale);
    default:
        FATAL() << "Unsupported voxel format";
    };

    return stk::Volume();
}


stk::Volume filters::downsample_vectorfield(const stk::Volume& vol, float scale, stk::Volume& residual)
{
    ASSERT(scale > 0.0f && scale < 1.0f);
    FATAL_IF(vol.voxel_type() != stk::Type_Float3)
        << "Unsupported voxel format";

    if (vol.voxel_type() == stk::Type_Float3) {
        stk::VolumeHelper<float3> field(vol);
        float inv_scale = 1.0f / scale;

        dim3 old_dims = field.size();
        dim3 new_dims {
            uint32_t(ceil(old_dims.x * scale)),
            uint32_t(ceil(old_dims.y * scale)),
            uint32_t(ceil(old_dims.z * scale))
        };

        stk::VolumeHelper<float3> result(new_dims);
        result.set_origin(field.origin());

        float3 old_spacing = field.spacing();
        float3 new_spacing {
            old_spacing.x * (old_dims.x / float(new_dims.x)),
            old_spacing.y * (old_dims.y / float(new_dims.y)),
            old_spacing.z * (old_dims.z / float(new_dims.z))
        };
        result.set_spacing(new_spacing);

        #pragma omp parallel for
        for (int z = 0; z < int(new_dims.z); ++z) {
            for (int y = 0; y < int(new_dims.y); ++y) {
                for (int x = 0; x < int(new_dims.x); ++x) {
                    int px = int(x * inv_scale);
                    int py = int(y * inv_scale);
                    int pz = int(z * inv_scale);

                    float3 v = field(px, py, pz);

                    v = v + field.at(px+1, py, pz, stk::Border_Replicate);
                    v = v + field.at(px, py+1, pz, stk::Border_Replicate);
                    v = v + field.at(px, py, pz+1, stk::Border_Replicate);
                    v = v + field.at(px+1, py+1, pz, stk::Border_Replicate);
                    v = v + field.at(px+1, py, pz+1, stk::Border_Replicate);
                    v = v + field.at(px, py+1, pz+1, stk::Border_Replicate);
                    v = v + field.at(px+1, py+1, pz+1, stk::Border_Replicate);
                    
                    float s = 1.0f / 8.0f;
                    result(x, y, z) = float3{s*v.x, s*v.y, s*v.z};
                }
            }
        }
        
	    stk::VolumeHelper<float3> tmp(field.size());
    
        #pragma omp parallel for
        for (int z = 0; z < int(old_dims.z); ++z) {
            for (int y = 0; y < int(old_dims.y); ++y) {
                for (int x = 0; x < int(old_dims.x); ++x) {
                    tmp(x, y, z) = field(x, y, z) - 
                        result.linear_at(scale*x, scale*y, scale*z, stk::Border_Replicate);
                }
            }
        }
        residual = tmp;
        return result;
    }
    return stk::Volume();
}
stk::Volume filters::upsample_vectorfield(
    const stk::Volume& vol, 
    const dim3& new_dims, 
    const stk::Volume& residual)
{
    FATAL_IF(vol.voxel_type() != stk::Type_Float3)
        << "Unsupported voxel format";
    
    if (vol.voxel_type() == stk::Type_Float3) {
        stk::VolumeFloat3 field(vol);

        dim3 old_dims = field.size();
        float3 inv_scale{
            float(old_dims.x) / new_dims.x,
            float(old_dims.y) / new_dims.y,
            float(old_dims.z) / new_dims.z
        };

        stk::VolumeFloat3 out(new_dims);
        out.set_origin(field.origin());

        float3 old_spacing = field.spacing();
        float3 new_spacing{
            old_spacing.x * inv_scale.x,
            old_spacing.y * inv_scale.y,
            old_spacing.z * inv_scale.z
        };
        out.set_spacing(new_spacing);

        if (residual.valid())
        {
            ASSERT(out.size() == residual.size());
            FATAL_IF(residual.voxel_type() != stk::Type_Float3)
                << "Unsupported voxel format";

            stk::VolumeFloat3 residual_float3(residual);
            
            #pragma omp parallel for
            for (int z = 0; z < int(new_dims.z); ++z) {
                for (int y = 0; y < int(new_dims.y); ++y) {
                    for (int x = 0; x < int(new_dims.x); ++x) {
                        float3 d = field.linear_at(
                                    inv_scale.x*x, 
                                    inv_scale.y*y, 
                                    inv_scale.z*z, 
                                    stk::Border_Replicate) 
                            + residual_float3(x, y, z);
                        out(x, y, z) = d; 
                    }
                }
            }

        }
        else
        {
            #pragma omp parallel for
            for (int z = 0; z < int(new_dims.z); ++z) {
                for (int y = 0; y < int(new_dims.y); ++y) {
                    for (int x = 0; x < int(new_dims.x); ++x) {
                        float3 d = field.linear_at(
                            inv_scale.x*x, 
                            inv_scale.y*y, 
                            inv_scale.z*z, 
                            stk::Border_Replicate);
                        out(x, y, z) = d; 
                    }
                }
            }
        }
        return out;
    }
    return stk::Volume();
}
