#include "resample.h"

#include "debug/assert.h"
#include "filters/gaussian_filter.h"
#include "math/float3.h"
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


Volume filters::downsample_vectorfield(const Volume& vol, float scale, Volume& residual)
{
    assert(scale > 0.0f && scale < 1.0f);
    assert(vol.voxel_type() == voxel::Type_Float3);
    if (vol.voxel_type() == voxel::Type_Float3)
    {
        VolumeHelper<float3> field(vol);
        float inv_scale = 1.0f / scale;

        Dims old_dims = field.size();
        Dims new_dims{
            uint32_t(ceil(old_dims.width * scale)),
            uint32_t(ceil(old_dims.height * scale)),
            uint32_t(ceil(old_dims.depth * scale))
        };

        VolumeHelper<float3> result(new_dims);
        result.set_origin(field.origin());

        float3 old_spacing = field.spacing();
        float3 new_spacing{
            old_spacing.x * inv_scale,
            old_spacing.y * inv_scale,
            old_spacing.z * inv_scale
        };
        result.set_spacing(new_spacing);

        #pragma omp parallel for
        for (int z = 0; z < int(new_dims.depth); ++z)
        {
            for (int y = 0; y < int(new_dims.height); ++y)
            {
                for (int x = 0; x < int(new_dims.width); ++x)
                {
                    int px = int(x * inv_scale);
                    int py = int(y * inv_scale);
                    int pz = int(z * inv_scale);

                    float3 v = field(px, py, pz);

                    v = v + field.at(px+1, py, pz, volume::Border_Replicate);
                    v = v + field.at(px, py+1, pz, volume::Border_Replicate);
                    v = v + field.at(px, py, pz+1, volume::Border_Replicate);
                    v = v + field.at(px+1, py+1, pz, volume::Border_Replicate);
                    v = v + field.at(px+1, py, pz+1, volume::Border_Replicate);
                    v = v + field.at(px, py+1, pz+1, volume::Border_Replicate);
                    v = v + field.at(px+1, py+1, pz+1, volume::Border_Replicate);
                    
                    float s = scale / 8.0f;
                    result(x, y, z) = float3{s*v.x, s*v.y, s*v.z};
                }
            }
        }
        
	    VolumeHelper<float3> tmp(field.size());
    
        #pragma omp parallel for
        for (int z = 0; z < int(old_dims.depth); ++z)
        {
            for (int y = 0; y < int(old_dims.height); ++y)
            {
                for (int x = 0; x < int(old_dims.width); ++x)
                {
                    tmp(x, y, z) = field(x, y, z) - 
                        inv_scale * result.linear_at(scale*x, scale*y, scale*z, volume::Border_Replicate);
                }
            }
        }
        residual = tmp;
        return result;
    }
    return Volume();
}
Volume filters::upsample_vectorfield(const Volume& vol, const Dims& new_dims, const Volume& residual)
{
    assert(vol.voxel_type() == voxel::Type_Float3);
    if (vol.voxel_type() == voxel::Type_Float3)
    {
        VolumeFloat3 field(vol);

        Dims old_dims = field.size();
        float3 scale{
            new_dims.width / float(old_dims.width),
            new_dims.height / float(old_dims.height),
            new_dims.depth / float(old_dims.depth)
        };
        float3 inv_scale{
            float(old_dims.width) / new_dims.width,
            float(old_dims.height) / new_dims.height,
            float(old_dims.depth) / new_dims.depth
        };

        VolumeFloat3 out(new_dims);
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
            assert(residual.voxel_type() == voxel::Type_Float3);
            assert(out.size() == residual.size());

            VolumeFloat3 residual_float3(residual);
            
            #pragma omp parallel for
            for (int z = 0; z < int(new_dims.depth); ++z)
            {
                for (int y = 0; y < int(new_dims.height); ++y)
                {
                    for (int x = 0; x < int(new_dims.width); ++x)
                    {
                        float3 d = field.linear_at(inv_scale.x*x, inv_scale.y*y, inv_scale.z*z, volume::Border_Replicate) 
                            + residual_float3(x, y, z);
                        out(x, y, z) = {scale.x * d.x, scale.y * d.y, scale.z * d.z}; 
                    }
                }
            }

        }
        else
        {
            #pragma omp parallel for
            for (int z = 0; z < int(new_dims.depth); ++z)
            {
                for (int y = 0; y < int(new_dims.height); ++y)
                {
                    for (int x = 0; x < int(new_dims.width); ++x)
                    {
                        float3 d = field.linear_at(inv_scale.x*x, inv_scale.y*y, inv_scale.z*z, volume::Border_Replicate);
                        out(x, y, z) = {scale.x * d.x, scale.y * d.y, scale.z * d.z};
                    }
                }
            }
        }
        return out;
    }
    return Volume();
}
