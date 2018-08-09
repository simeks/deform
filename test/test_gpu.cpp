#include "catch.hpp"

#ifdef DF_USE_CUDA

#include <deform_lib/filters/gaussian_filter.h>
#include <deform_lib/filters/gpu/gaussian_filter.h>

#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

TEST_CASE("gpu_gaussian_filter", "")
{
    stk::VolumeFloat src({16,16,16});

    for (int z = 0; z < (int)src.size().z; ++z) 
    for (int y = 0; y < (int)src.size().y; ++y) 
    for (int x = 0; x < (int)src.size().x; ++x) {
        
        int r = (x-7)*(x-7) + (y-7)*(y-7) + (z-7)*(z-7);
        if (r < 6) {
            src(x,y,z) = 1.0f;
        }
        else {
            src(x,y,z) = 0.0f;            
        }
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat out = filters::gaussian_filter_3d(src, 2.5f);

    stk::GpuVolume gpu_src(src);
    stk::VolumeFloat gpu_out = filters::gpu::gaussian_filter_3d(gpu_src, 2.5f).download();

    for (int z = 0; z < (int)src.size().z; ++z) 
    for (int y = 0; y < (int)src.size().y; ++y) 
    for (int x = 0; x < (int)src.size().x; ++x) {
        REQUIRE(gpu_out(x,y,z) == Approx(out(x,y,z)));
    }
}

TEST_CASE("gpu_resample", "")
{

}
#endif // DF_USE_CUDA
