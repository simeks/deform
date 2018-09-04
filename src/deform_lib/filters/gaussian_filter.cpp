#include "gaussian_filter.h"

#include <stk/common/error.h>
#include <stk/image/volume.h>
#include <stk/math/float3.h>

#include <algorithm>
#include <math.h>

namespace
{
    template<typename TVoxelType>
    stk::VolumeHelper<TVoxelType> gaussian_filter_3d(const stk::VolumeHelper<TVoxelType>& img, float sigma)
    {
        stk::VolumeHelper<TVoxelType> result = img.clone(); // Clone to get size and meta data

        if (sigma <= 0.0f) {
            return result;
        }

        stk::VolumeHelper<TVoxelType> tmp = img.clone();

        double spacing_x = double(img.spacing().x);
        double spacing_y = double(img.spacing().y);
        double spacing_z = double(img.spacing().z);

        double factor = (-1.0) / (2 * sigma*sigma);

        dim3 dims = img.size();

        // X dimension
        int size = (int)ceil(3 * sigma / spacing_x);
        #pragma omp parallel for
        for (int y = 0; y < int(dims.y); ++y)
        {
            for (int z = 0; z < int(dims.z); ++z)
            {
                for (int x = 0; x < int(dims.x); ++x)
                {
                    TVoxelType value{0};
                    double norm = 0;
                    for (int t = -size; t < size + 1; t++)
                    {
                        double c = exp(factor*(t*spacing_x)*(t*spacing_x));

                        int sx = std::min(std::max<int>(x + t, 0), int(dims.x - 1));
                        value += TVoxelType(c*tmp(sx, y, z));
                        norm += c;
                    }
                    result(x, y, z) = TVoxelType(value / norm);
                }
            }
        }

        tmp.copy_from(result);

        //Y dimension
        size = (int)ceil(3 * sigma / spacing_y);
        #pragma omp parallel for
        for (int x = 0; x < int(dims.x); ++x)
        {
            for (int z = 0; z < int(dims.z); ++z)
            {
                for (int y = 0; y < int(dims.y); ++y)
                {
                    TVoxelType value{0};
                    double norm = 0;
                    for (int t = -size; t < size + 1; t++)
                    {
                        double c = exp(factor*(t*spacing_y)*(t*spacing_y));

                        int sy = std::min(std::max(y + t, 0), int(dims.y - 1));
                        value += TVoxelType(c*tmp(x, sy, z));
                        norm += c;
                    }
                    result(x, y, z) = TVoxelType(value / norm);
                }
            }
        }

        tmp.copy_from(result);
        
        //Z dimension
        size = (int)ceil(3 * sigma / spacing_z);
        #pragma omp parallel for
        for (int x = 0; x < int(dims.x); ++x)
        {
            for (int y = 0; y < int(dims.y); ++y)
            {
                for (int z = 0; z < int(dims.z); ++z)
                {
                    TVoxelType value{0};
                    double norm = 0;
                    for (int t = -size; t < size + 1; t++)
                    {
                        double c = exp(factor*(t*spacing_z)*(t*spacing_z));

                        int sz = std::min(std::max(z + t, 0), int(dims.z - 1));
                        value += TVoxelType(c*tmp(x, y, sz));
                        norm += c;
                    }
                    result(x, y, z) = TVoxelType(value / norm);
                }
            }
        }
        return result;
    }
}

stk::Volume filters::gaussian_filter_3d(const stk::Volume& volume, float sigma)
{
    switch (volume.voxel_type())
    {
    case stk::Type_Float:
        return ::gaussian_filter_3d<float>(volume, sigma);
    case stk::Type_Double:
        return ::gaussian_filter_3d<double>(volume, sigma);
    case stk::Type_Float3:
        return ::gaussian_filter_3d<float3>(volume, sigma);
    default:
        FATAL() << "Unsupported voxel format";
    };
    return stk::Volume();
}
