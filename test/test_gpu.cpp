#include "catch.hpp"

#ifdef DF_USE_CUDA

#include <deform_lib/filters/resample.h>
#include <deform_lib/filters/gpu/resample.h>

#include <deform_lib/filters/gaussian_filter.h>
#include <deform_lib/filters/gpu/gaussian_filter.h>

#include <deform_lib/registration/transform.h>
#include <deform_lib/registration/gpu/transform.h>

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
    
    CHECK(gpu_out.spacing().x == Approx(out.spacing().x));
    CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
    CHECK(gpu_out.spacing().z == Approx(out.spacing().z));

    CHECK(gpu_out.origin().x == Approx(out.origin().x));
    CHECK(gpu_out.origin().y == Approx(out.origin().y));
    CHECK(gpu_out.origin().z == Approx(out.origin().z));

    for (int z = 0; z < (int)src.size().z; ++z) 
    for (int y = 0; y < (int)src.size().y; ++y) 
    for (int x = 0; x < (int)src.size().x; ++x) {
        REQUIRE(gpu_out(x,y,z) == Approx(out(x,y,z)));
    }
}

TEST_CASE("gpu_downsample_gaussian", "")
{
    stk::VolumeFloat src({16,16,16});

    for (int z = 0; z < (int)src.size().z; ++z) 
    for (int y = 0; y < (int)src.size().y; ++y) 
    for (int x = 0; x < (int)src.size().x; ++x) {
        
        int r = (x-7)*(x-7) + (y-7)*(y-7) + (z-7)*(z-7);
        if (r < 6) {
            src(x,y,z) = 10.0f;
        }
        else {
            src(x,y,z) = 1.0f;            
        }
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat out = filters::downsample_volume_by_2(src);

    stk::GpuVolume gpu_src(src);
    stk::VolumeFloat gpu_out = filters::gpu::downsample_volume_by_2(gpu_src).download();

    CHECK(gpu_out.spacing().x == Approx(out.spacing().x));
    CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
    CHECK(gpu_out.spacing().z == Approx(out.spacing().z));

    CHECK(gpu_out.origin().x == Approx(out.origin().x));
    CHECK(gpu_out.origin().y == Approx(out.origin().y));
    CHECK(gpu_out.origin().z == Approx(out.origin().z));

    for (int z = 0; z < (int)out.size().z; ++z) 
    for (int y = 0; y < (int)out.size().y; ++y) 
    for (int x = 0; x < (int)out.size().x; ++x) {
        REQUIRE(gpu_out(x,y,z) == Approx(out(x,y,z)));
    }
}

TEST_CASE("gpu_downsample_vectorfield", "")
{
    stk::VolumeFloat3 src({16,16,16});
    stk::VolumeFloat4 src4({16,16,16}); // CUDA do not support float3 so we add an empty channel

    for (int z = 0; z < (int)src.size().z; ++z) 
    for (int y = 0; y < (int)src.size().y; ++y) 
    for (int x = 0; x < (int)src.size().x; ++x) {
        
        int r = (x-7)*(x-7) + (y-7)*(y-7) + (z-7)*(z-7);
        if (r < 6) {
            src(x,y,z) = {2.0f, 1.0f, 0.5f};
        }
        else {
            src(x,y,z) = {1.0f, 1.0f, 1.0f};
        }
        src4(x,y,z) = {src(x,y,z).x, src(x,y,z).y, src(x,y,z).z, 0.0f};
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat3 out = filters::downsample_vectorfield_by_2(src);

    stk::GpuVolume gpu_src(src4);
    stk::VolumeFloat4 gpu_out = filters::gpu::downsample_vectorfield_by_2(gpu_src).download();

    CHECK(gpu_out.spacing().x == Approx(out.spacing().x));
    CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
    CHECK(gpu_out.spacing().z == Approx(out.spacing().z));

    CHECK(gpu_out.origin().x == Approx(out.origin().x));
    CHECK(gpu_out.origin().y == Approx(out.origin().y));
    CHECK(gpu_out.origin().z == Approx(out.origin().z));

    for (int z = 0; z < (int)out.size().z; ++z) 
    for (int y = 0; y < (int)out.size().y; ++y) 
    for (int x = 0; x < (int)out.size().x; ++x) {
        REQUIRE(gpu_out(x,y,z).x == Approx(out(x,y,z).x));
        REQUIRE(gpu_out(x,y,z).y == Approx(out(x,y,z).y));
        REQUIRE(gpu_out(x,y,z).z == Approx(out(x,y,z).z));
        REQUIRE(gpu_out(x,y,z).w == Approx(0.0f));
    }
}

TEST_CASE("gpu_upsample_vectorfield", "")
{
    stk::VolumeFloat3 src({8,8,8});
    stk::VolumeFloat4 src4({8,8,8}); // CUDA do not support float3 so we add an empty channel

    for (int z = 0; z < (int)src.size().z; ++z) 
    for (int y = 0; y < (int)src.size().y; ++y) 
    for (int x = 0; x < (int)src.size().x; ++x) {
        int r = (x-3)*(x-3) + (y-3)*(y-3) + (z-3)*(z-3);
        if (r < 6) {
            src(x,y,z) = {2.0f, 1.0f, 0.5f};
        }
        else {
            src(x,y,z) = {1.0f, 1.0f, 1.0f};
        }
        src4(x,y,z) = {src(x,y,z).x, src(x,y,z).y, src(x,y,z).z, 0.0f};
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat3 out = filters::upsample_vectorfield(src, {16,16,16});

    stk::GpuVolume gpu_src(src4);
    stk::VolumeFloat4 gpu_out = filters::gpu::upsample_vectorfield(gpu_src, {16,16,16}).download();

    CHECK(gpu_out.spacing().x == Approx(out.spacing().x));
    CHECK(gpu_out.spacing().y == Approx(out.spacing().y));
    CHECK(gpu_out.spacing().z == Approx(out.spacing().z));

    CHECK(gpu_out.origin().x == Approx(out.origin().x));
    CHECK(gpu_out.origin().y == Approx(out.origin().y));
    CHECK(gpu_out.origin().z == Approx(out.origin().z));

    for (int z = 0; z < (int)out.size().z; ++z) 
    for (int y = 0; y < (int)out.size().y; ++y) 
    for (int x = 0; x < (int)out.size().x; ++x) {
        REQUIRE(gpu_out(x,y,z).x == Approx(out(x,y,z).x));
        REQUIRE(gpu_out(x,y,z).y == Approx(out(x,y,z).y));
        REQUIRE(gpu_out(x,y,z).z == Approx(out(x,y,z).z));
        REQUIRE(gpu_out(x,y,z).w == Approx(0.0f));
    }
}


TEST_CASE("gpu_transform", "")
{
    stk::VolumeFloat src({8,8,8});
    src.set_origin(float3{1, 2, 3});
    src.set_spacing(float3{1.1f, 1.2f, 1.3f});
    stk::VolumeFloat3 def3({8,8,8}); // CUDA do not support float3 so we add an empty channel
    stk::VolumeFloat4 def4({8,8,8}); // CUDA do not support float3 so we add an empty channel

    for (int z = 0; z < (int)src.size().z; ++z) 
    for (int y = 0; y < (int)src.size().y; ++y) 
    for (int x = 0; x < (int)src.size().x; ++x) {
        int r = (x-3)*(x-3) + (y-3)*(y-3) + (z-3)*(z-3);
        if (r < 6) {
            src(x,y,z) = float(r);
        }
        else {
            src(x,y,z) = 1.0f;
        }

        def3(x,y,z) = {1.0f, 2.0f, 3.0f};
        def4(x,y,z) = {1.0f, 2.0f, 3.0f, 0.0f};
    }

    // Use CPU-version as ground truth
    stk::VolumeFloat out_lin = transform_volume(src, def3, transform::Interp_Linear);
    stk::VolumeFloat out_nn = transform_volume(src, def3, transform::Interp_NN);

    stk::GpuVolume gpu_src(src);
    stk::GpuVolume gpu_def(def4);
    stk::VolumeFloat gpu_out_lin = gpu::transform_volume(gpu_src, gpu_def, transform::Interp_Linear)
        .download();
    stk::VolumeFloat gpu_out_nn = gpu::transform_volume(gpu_src, gpu_def, transform::Interp_NN)
        .download();

    CHECK(gpu_out_lin.spacing().x == Approx(out_lin.spacing().x));
    CHECK(gpu_out_lin.spacing().y == Approx(out_lin.spacing().y));
    CHECK(gpu_out_lin.spacing().z == Approx(out_lin.spacing().z));

    CHECK(gpu_out_lin.origin().x == Approx(out_lin.origin().x));
    CHECK(gpu_out_lin.origin().y == Approx(out_lin.origin().y));
    CHECK(gpu_out_lin.origin().z == Approx(out_lin.origin().z));

    CHECK(gpu_out_nn.spacing().x == Approx(out_nn.spacing().x));
    CHECK(gpu_out_nn.spacing().y == Approx(out_nn.spacing().y));
    CHECK(gpu_out_nn.spacing().z == Approx(out_nn.spacing().z));

    CHECK(gpu_out_nn.origin().x == Approx(out_nn.origin().x));
    CHECK(gpu_out_nn.origin().y == Approx(out_nn.origin().y));
    CHECK(gpu_out_nn.origin().z == Approx(out_nn.origin().z));

    for (int z = 0; z < (int)out_nn.size().z; ++z) 
    for (int y = 0; y < (int)out_nn.size().y; ++y) 
    for (int x = 0; x < (int)out_nn.size().x; ++x) {
        REQUIRE(gpu_out_lin(x,y,z) == Approx(out_lin(x,y,z)));
        REQUIRE(gpu_out_nn(x,y,z) == Approx(out_nn(x,y,z)));
    }
}



#endif // DF_USE_CUDA
