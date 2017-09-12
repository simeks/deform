#include "transform.h"

#include <framework/debug/log.h>
#include <framework/math/float3.h>

namespace
{
    template<typename TVoxelType>
    VolumeHelper<TVoxelType> transform_volume(
        const VolumeHelper<TVoxelType>& src,
        const VolumeFloat3& def)
    {
        // Transformed volume will have the same dimensions and properties (origin and spacing) 
        //  as the deformation field and not the source image. The deformation field is inherited
        //  from the fixed volume in the registration and we want to transform the moving image (src)
        //  to the fixed image space.

        // There should be no requirements on src to have the same size, spacing and origin as
        //  the fixed image.

        Dims dims = def.size();

        VolumeHelper<TVoxelType> out(dims);
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
        for (int z = 0; z < int(dims.depth); ++z)
        {
            for (int y = 0; y < int(dims.height); ++y)
            {
                for (int x = 0; x < int(dims.width); ++x)
                {
                    // [fixed] -> [world] -> [moving]
                    float3 fixed_p = float3{float(x), float(y), float(z)} + def(x, y, z);
                    float3 moving_p = (fixed_p * fixed_spacing + fixed_origin - moving_origin) 
                        * inv_moving_spacing;

                    out(x, y, z) = src.linear_at(moving_p.x, moving_p.y, moving_p.z, volume::Border_Constant);
                }
            }
        }
        return out;
    }
}

Volume transform_volume(const Volume& src, const VolumeFloat3& def)
{
    if (src.voxel_type() == voxel::Type_Float)
    {
        return transform_volume<float>(src, def);
    }
    else if (src.voxel_type() == voxel::Type_Double)
    {
        return transform_volume<double>(src, def);
    }
    else
    {
        LOG(Error, "transform_volume: Unsupported volume type (type: %d)\n", src.voxel_type());
    }
    return Volume();
}
