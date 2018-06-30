#include "jacobian.h"

#include <stk/math/float3.h>

stk::Volume calculate_jacobian(const stk::VolumeFloat3& def)
{
    dim3 dims = def.size();

    stk::VolumeHelper<double> out(dims);
    out.set_origin(def.origin());
    out.set_spacing(def.spacing());

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
                double a = def_dx.x;
                double b = def_dy.x;
                double c = def_dz.x;

                double d = def_dx.y;
                double e = def_dy.y;
                double f = def_dz.y;

                double g = def_dx.z;
                double h = def_dy.z;
                double i = def_dz.z;

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

