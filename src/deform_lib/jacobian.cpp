#include "jacobian.h"

#include <stk/math/float3.h>

stk::Volume calculate_jacobian(const stk::VolumeFloat3& def)
{
    dim3 dims = def.size();

    stk::VolumeHelper<JAC_TYPE> out(dims);
    out.copy_meta_from(def);

    long W = dims.x;
    long H = dims.y;
    long D = dims.z;

    stk::BorderMode border_mode = stk::Border_Replicate;

    float3 spacing = def.spacing();
    float3 spacing_inv{ 1 / spacing.x, 1 / spacing.y, 1 / spacing.z };

    #pragma omp parallel for
    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) {
            for (int z = 0; z < D; z++) {
                float3 def_dx = 0.5f * spacing_inv.x * (def.at(x+1,y,z, border_mode) - def.at(x-1,y,z, border_mode));
                float3 def_dy = 0.5f * spacing_inv.y * (def.at(x,y+1,z, border_mode) - def.at(x,y-1,z, border_mode));
                float3 def_dz = 0.5f * spacing_inv.z * (def.at(x,y,z+1, border_mode) - def.at(x,y,z-1, border_mode));

                //Partial derivatives
                JAC_TYPE a = def_dx.x;
                JAC_TYPE b = def_dy.x;
                JAC_TYPE c = def_dz.x;

                JAC_TYPE d = def_dx.y;
                JAC_TYPE e = def_dy.y;
                JAC_TYPE f = def_dz.y;

                JAC_TYPE g = def_dx.z;
                JAC_TYPE h = def_dy.z;
                JAC_TYPE i = def_dz.z;

                // Compose with the identity transform
                a += 1;
                e += 1;
                i += 1;

                out(x, y, z) = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h;
            }
        }
    }

    return out;
}

