#include "resample.h"

#include "gaussian_filter.h"

#include <stk/common/assert.h>
#include <stk/image/volume.h>
#include <stk/math/float3.h>

#include <algorithm>

namespace
{
    template<typename TVoxelType>
    stk::VolumeHelper<TVoxelType> downsample_volume_by_2(const stk::VolumeHelper<TVoxelType>& src)
    {
        dim3 old_dims = src.size();
        dim3 new_dims {
            uint32_t(ceil(old_dims.x * 0.5f)),
            uint32_t(ceil(old_dims.y * 0.5f)),
            uint32_t(ceil(old_dims.z * 0.5f)),
        };

        stk::VolumeHelper<TVoxelType> dest(new_dims);
        dest.copy_meta_from(src);

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
                    dest(x, y, z) = src(2*x, 2*y, 2*z);
                }
            }
        }
        return dest;
    }
}


stk::Volume filters::downsample_volume_by_2(const stk::Volume& vol)
{
    FATAL_IF(vol.voxel_type() != stk::Type_Float &&
             vol.voxel_type() != stk::Type_Double)
        << "Unsupported voxel format";

    float3 spacing = vol.spacing();
    float unit_sigma = std::min(spacing.x, std::min(spacing.y, spacing.z));
    stk::Volume filtered = filters::gaussian_filter_3d(vol, unit_sigma);

    switch (vol.voxel_type()) {
    case stk::Type_Float:
        return ::downsample_volume_by_2<float>(filtered);
    case stk::Type_Double:
        return ::downsample_volume_by_2<double>(filtered);
    default:
        FATAL() << "Unsupported voxel format";
    };

    return stk::Volume();
}

stk::Volume filters::downsample_vectorfield_by_2(const stk::Volume& field
#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    , stk::Volume& residual
#endif
)
{
    FATAL_IF(field.voxel_type() != stk::Type_Float3)
        << "Unsupported voxel format";

    float3 spacing = field.spacing();
    float unit_sigma = std::min(spacing.x, std::min(spacing.y, spacing.z));
    stk::Volume filtered = filters::gaussian_filter_3d(field, unit_sigma);

    stk::VolumeFloat3 result = ::downsample_volume_by_2<float3>(filtered);

    #ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
        dim3 old_dims = field.size();
        stk::VolumeFloat3 src(field);
        stk::VolumeFloat3 tmp(old_dims);


        #pragma omp parallel for
        for (int z = 0; z < int(old_dims.z); ++z) {
            for (int y = 0; y < int(old_dims.y); ++y) {
                for (int x = 0; x < int(old_dims.x); ++x) {
                    tmp(x, y, z) = src(x, y, z) -
                        result.linear_at(0.5f*x, 0.5f*y, 0.5f*z, stk::Border_Replicate);
                }
            }
        }
        residual = tmp;
    #endif
    return result;
}

stk::Volume filters::upsample_vectorfield(const stk::Volume& vol, const dim3& new_dims
#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
    , const stk::Volume& residual
#endif
)
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
        out.copy_meta_from(field);

        float3 old_spacing = field.spacing();
        float3 new_spacing{
            old_spacing.x * inv_scale.x,
            old_spacing.y * inv_scale.y,
            old_spacing.z * inv_scale.z
        };
        out.set_spacing(new_spacing);

#ifdef DF_ENABLE_DISPLACEMENT_FIELD_RESIDUALS
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
#endif
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
