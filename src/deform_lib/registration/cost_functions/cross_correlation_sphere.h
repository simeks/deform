#pragma once

#include "sub_function.h"

template<typename T>
struct NCCFunction_sphere : public SubFunction
{
    /*!
     * \brief Compute NCC with a spheric window.
     */
    NCCFunction_sphere(
                const stk::VolumeFloat3& df,
                const stk::VolumeHelper<T>& fixed,
                const stk::VolumeHelper<T>& moving,
                const int radius) :
        _fixed(fixed),
        _moving(moving),
        _radius(radius)
    {}


    virtual ~NCCFunction_sphere() {}


    float cost(const int3& p, const float3& dv)
    {
        const auto pt = _df.index2point(p);

        // [fixed] -> [world] -> [moving]
        const auto fixed_p = _fixed.point2index(pt);
        const auto moving_p = _moving.point2index(pt + dv);

        // Check whether the point is masked out
        float mask_value = 1.0f;
        if (_moving_mask.valid()) {
            mask_value = _moving_mask.linear_at(moving_p, stk::Border_Constant);
            if (mask_value <= std::numeric_limits<float>::epsilon()) {
                return 0.0f;
            }
        }

        // [Filip]: Addition for partial-body registrations
        if (moving_p.x < 0 || moving_p.x >= _moving.size().x ||
            moving_p.y < 0 || moving_p.y >= _moving.size().y ||
            moving_p.z < 0 || moving_p.z >= _moving.size().z) {
            return 0;
        }

        double sff = 0.0;
        double smm = 0.0;
        double sfm = 0.0;
        double sf = 0.0;
        double sm = 0.0;
        size_t n = 0;

        for (int dz = -_radius; dz <= _radius; ++dz) {
            for (int dy = -_radius; dy <= _radius; ++dy) {
                for (int dx = -_radius; dx <= _radius; ++dx) {
                    // TODO: Does not account for anisotropic volumes
                    int r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 > _radius * _radius)
                        continue;

                    //int3 fp{p.x + dx, p.y + dy, p.z + dz};

                    //if (!stk::is_inside(_fixed.size(), fp))
                    //    continue;

                    float3 fp{fixed_p.x + dx, fixed_p.y + dy, fixed_p.z + dz};
                    float3 mp{moving_p.x + dx, moving_p.y + dy, moving_p.z + dz};

                    //T fixed_v = _fixed(fp);
                    T fixed_v = _fixed.linear_at(fp, stk::Border_Constant);
                    T moving_v = _moving.linear_at(mp, stk::Border_Constant);

                    sff += fixed_v * fixed_v;
                    smm += moving_v * moving_v;
                    sfm += fixed_v*moving_v;
                    sm += moving_v;
                    sf += fixed_v;

                    ++n;
                }
            }
        }

        if (n == 0)
            return 0.0f;

        // Subtract mean
        sff -= (sf * sf / n);
        smm -= (sm * sm / n);
        sfm -= (sf * sm / n);

        double d = sqrt(sff*smm);

        if(d > 1e-5) {
            return mask_value * float(0.5*(1.0-sfm / d));
        }
        return 0.0f;
    }

    stk::VolumeFloat3 _df;
    stk::VolumeHelper<T> _fixed;
    stk::VolumeHelper<T> _moving;

    const int _radius;
};

