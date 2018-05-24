#include "transform.h"

#include <stk/common/log.h>
#include <stk/math/float3.h>

namespace
{
    template<typename TVoxelType>
    stk::VolumeHelper<TVoxelType> transform_volume_nn(
        const stk::VolumeHelper<TVoxelType>& src,
        const stk::VolumeFloat3& def)
    {
        // Transformed volume will have the same dimensions and properties (origin and spacing) 
        //  as the deformation field and not the source image. The deformation field is inherited
        //  from the fixed volume in the registration and we want to transform the moving image (src)
        //  to the fixed image space.

        // There should be no requirements on src to have the same size, spacing and origin as
        //  the fixed image.

        dim3 dims = def.size();

        stk::VolumeHelper<TVoxelType> out(dims);
        out.set_origin(def.origin());
        out.set_spacing(def.spacing());

        float3 fixed_origin = def.origin();
        float3 moving_origin = src.origin();
        
        float3 fixed_spacing = def.spacing();
        float3 inv_moving_spacing{
            1.0f / src.spacing().x,
            1.0f / src.spacing().y,
            1.0f / src.spacing().z
        };

        #pragma omp parallel for
        for (int z = 0; z < int(dims.z); ++z) {
            for (int y = 0; y < int(dims.y); ++y) {
                for (int x = 0; x < int(dims.x); ++x) {
                    // [fixed] -> [world] -> [moving]
                    float3 fixed_p = float3{float(x), float(y), float(z)} + def(x, y, z);
                    float3 moving_p = (fixed_p * fixed_spacing + fixed_origin - moving_origin) 
                        * inv_moving_spacing;

                    out(x, y, z) = src.at(
                        int(round(moving_p.x)), int(round(moving_p.y)), int(round(moving_p.z)), 
                        stk::Border_Constant);
                }
            }
        }
        return out;
    }

    template<typename TVoxelType>
    stk::VolumeHelper<TVoxelType> transform_volume_linear(
        const stk::VolumeHelper<TVoxelType>& src,
        const stk::VolumeFloat3& def)
    {
        // Transformed volume will have the same dimensions and properties (origin and spacing) 
        //  as the deformation field and not the source image. The deformation field is inherited
        //  from the fixed volume in the registration and we want to transform the moving image (src)
        //  to the fixed image space.

        // There should be no requirements on src to have the same size, spacing and origin as
        //  the fixed image.

        dim3 dims = def.size();

        stk::VolumeHelper<TVoxelType> out(dims);
        out.set_origin(def.origin());
        out.set_spacing(def.spacing());

        float3 fixed_origin = def.origin();
        float3 moving_origin = src.origin();
        
        float3 fixed_spacing = def.spacing();
        float3 inv_moving_spacing{
            1.0f / src.spacing().x,
            1.0f / src.spacing().y,
            1.0f / src.spacing().z
        };

        #pragma omp parallel for
        for (int z = 0; z < int(dims.z); ++z) {
            for (int y = 0; y < int(dims.y); ++y) {
                for (int x = 0; x < int(dims.x); ++x) {
                    // [fixed] -> [world] -> [moving]
                    float3 fixed_p = float3{float(x), float(y), float(z)} + def(x, y, z);
                    float3 moving_p = (fixed_p * fixed_spacing + fixed_origin - moving_origin) 
                        * inv_moving_spacing;

                    out(x, y, z) = src.linear_at(moving_p.x, moving_p.y, moving_p.z, stk::Border_Constant);
                }
            }
        }
        return out;
    }
}

stk::Volume transform_volume(const stk::Volume& src, const stk::VolumeFloat3& def, transform::Interp interp)
{
    if (interp == transform::Interp_NN) {
        if (src.voxel_type() == stk::Type_Float) {
            return transform_volume_nn<float>(src, def);
        }
        else if (src.voxel_type() == stk::Type_Double) {
            return transform_volume_nn<double>(src, def);
        }
        else if (src.voxel_type() == stk::Type_UChar) {
            return transform_volume_nn<uint8_t>(src, def);
        }
        else {
            LOG(Error) << "transform_volume: Unsupported volume type (type: " << src.voxel_type() << ")";
        }
    }
    else if (interp == transform::Interp_Linear) {
        if (src.voxel_type() == stk::Type_Float) {
            return transform_volume_linear<float>(src, def);
        }
        else if (src.voxel_type() == stk::Type_Double) {
            return transform_volume_linear<double>(src, def);
        }
        else {
            LOG(Error) << "transform_volume: Unsupported volume type (type: " << src.voxel_type() << ")";
        }
    }
    else {
        LOG(Error) << "transform_volume: Unsupported interpolation method (given: " << interp << ")";
    }
    return stk::Volume();
}
