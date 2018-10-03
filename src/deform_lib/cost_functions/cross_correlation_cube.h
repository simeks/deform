#pragma once

#include "sub_function.h"

#if DF_USE_ISPC

#include <ispc_lib.h>

template<typename T>
struct NCCFunction_cube : public SubFunction
{
    /*!
     * \brief Compute NCC with a cubic window of side `2 × radius + 1`.
     *
     * The code of this function is vectorized with ISPC.
     */
    NCCFunction_cube(const stk::VolumeHelper<T>& fixed,
                     const stk::VolumeHelper<T>& moving,
                     const int radius) :
        _fixed({fixed.size().x + 2*radius,
                fixed.size().y + 2*radius,
                fixed.size().z + 2*radius},
               0.0f),
        _moving({moving.size().x + 2,
                 moving.size().y + 2,
                 moving.size().z + 2},
                0.0f),
        _radius(radius)
    {
        // fixed volume, zero-padding with thickness `radius`
        for (int z = 0; z < (int) fixed.size().z; ++z) {
            for (int y = 0; y < (int) fixed.size().y; ++y) {
                for (int x = 0; x < (int) fixed.size().x; ++x) {
                    _fixed(radius + x, radius + y, radius + z) = float(fixed(x, y, z));
                }
            }
        }
        _fixed.set_origin(fixed.origin());
        _fixed.set_spacing(fixed.spacing());

        // moving volume, zero-padding with thickness 1
        for (int z = 0; z < (int) moving.size().z; ++z) {
            for (int y = 0; y < (int) moving.size().y; ++y) {
                for (int x = 0; x < (int) moving.size().x; ++x) {
                    _moving(1 + x, 1 + y, 1 + z) = float(moving(x, y, z));
                }
            }
        }
        _moving.set_origin(moving.origin());
        _moving.set_spacing(moving.spacing());
    }


    float cost(const int3& p, const float3& def)
    {
        // [fixed] -> [world] -> [moving]
        const auto moving_p = _moving.point2index(_fixed.index2point(p) + def);

        // [Filip]: Addition for partial-body registrations
        // NOTE: _moving.size() includes padding
        if (moving_p.x < 0 || moving_p.x >= _moving.size().x - 2 ||
            moving_p.y < 0 || moving_p.y >= _moving.size().y - 2 ||
            moving_p.z < 0 || moving_p.z >= _moving.size().z - 2) {
            return 0;
        }

        // Cast stuff to ispc types
        const ispc::int3 fp = {p.x + _radius, p.y + _radius, p.z + _radius};
        const ispc::float3 mp = {moving_p.x + 1.0f, moving_p.y + 1.0f, moving_p.z + 1.0f};
        const ispc::int3 fs = {
            (int) _fixed.size().x,
            (int) _fixed.size().y,
            (int) _fixed.size().z
        };
        const ispc::int3 ms = {
            (int) _moving.size().x,
            (int) _moving.size().y,
            (int) _moving.size().z
        };
        const float *fixed = (const float*) _fixed.ptr();
        const float *moving = (const float*) _moving.ptr();

        // Evaluate the ispc kernel
        return ispc::ncc(_radius, fp, mp, fixed, moving, fs, ms);
    }

    stk::VolumeHelper<float> _fixed;
    stk::VolumeHelper<float> _moving;
    const int _radius;
};

#else // DF_USE_ISPC

template<typename T>
struct NCCFunction_cube : public SubFunction
{
    NCCFunction_cube(const stk::VolumeHelper<T>& /* fixed */,
                     const stk::VolumeHelper<T>& /* moving */,
                     const int /* radius */)
    {
        throw std::runtime_error("deform_lib must be built with ISPC support "
                                 "in order to use NCC with cubic window.");
    }


    float cost(const int3& /* p */, const float3& /* def */)
    {
        throw std::runtime_error("deform_lib must be built with ISPC support "
                                 "in order to use NCC with cubic window.");
    }
};

#endif // DF_USE_ISPC

