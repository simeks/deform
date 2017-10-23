#include "debug/assert.h"
#include "gaussian_filter.h"
#include "volume/volume_helper.h"

#include <algorithm>
#include <math.h>

namespace
{
    template<typename TVoxelType>
    VolumeHelper<TVoxelType> gaussian_filter_3d(const VolumeHelper<TVoxelType>& img, float sigma)
    {
        VolumeHelper<TVoxelType> result = img.clone(); // Clone to get size and meta data
        VolumeHelper<TVoxelType> tmp = img.clone();

        double spacing_x = double(img.spacing().x);
        double spacing_y = double(img.spacing().y);
        double spacing_z = double(img.spacing().z);

        double factor = (-1.0) / (2 * sigma*sigma);

        Dims dims = img.size();

        // X dimension
        int size = (int)ceil(3 * sigma / spacing_x);
        #pragma omp parallel for
        for (int y = 0; y < int(dims.height); ++y)
        {
            for (int z = 0; z < int(dims.depth); ++z)
            {
                for (int x = 0; x < int(dims.width); ++x)
                {
                    TVoxelType value{0};
                    double norm = 0;
                    for (int t = -size; t < size + 1; t++)
                    {
                        double c = exp(factor*(t*spacing_x)*(t*spacing_x));

                        int sx = std::min(std::max<int>(x + t, 0), int(dims.width - 1));
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
        for (int x = 0; x < int(dims.width); ++x)
        {
            for (int z = 0; z < int(dims.depth); ++z)
            {
                for (int y = 0; y < int(dims.height); ++y)
                {
                    TVoxelType value{0};
                    double norm = 0;
                    for (int t = -size; t < size + 1; t++)
                    {
                        double c = exp(factor*(t*spacing_x)*(t*spacing_x));

                        int sy = std::min(std::max(y + t, 0), int(dims.height - 1));
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
        for (int x = 0; x < int(dims.width); ++x)
        {
            for (int y = 0; y < int(dims.height); ++y)
            {
                for (int z = 0; z < int(dims.depth); ++z)
                {
                    TVoxelType value{0};
                    double norm = 0;
                    for (int t = -size; t < size + 1; t++)
                    {
                        double c = exp(factor*(t*spacing_x)*(t*spacing_x));

                        int sz = std::min(std::max(z + t, 0), int(dims.depth - 1));
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

Volume filters::gaussian_filter_3d(const Volume& volume, float sigma)
{
    switch (volume.voxel_type())
    {
    case voxel::Type_Float:
        return ::gaussian_filter_3d<float>(volume, sigma);
    case voxel::Type_Double:
        return ::gaussian_filter_3d<double>(volume, sigma);
    default:
        assert(false);
    };
    return Volume();
}
