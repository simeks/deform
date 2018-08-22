#include "catch.hpp"

#ifdef DF_USE_CUDA

#include <deform_lib/cost_function.h>
#include <deform_lib/registration/gpu/cost_function.h>

#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include <random>


/*
    Using textures in CUDA has quite a big impact on precision as
    the coordinates are computed with only 8 bit fractional precision [1].
    Therefore these tests should have quite a large epsilon.

    TODO: Investigate performance impact of doing interpolation by ourself

    [1] Texture Fetching, CUDA C Programming Guide, https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching
*/

namespace {
    int _data_seed = 87575;
}

void fill_data(stk::VolumeFloat& d) {
    std::mt19937 gen(_data_seed);
    std::uniform_int_distribution<> dis(0, 10000);

    for (int z = 0; z < (int)d.size().z; ++z) {
    for (int y = 0; y < (int)d.size().y; ++y) {
    for (int x = 0; x < (int)d.size().x; ++x) {
        d(x,y,z) = (float)dis(gen);
    }
    }
    }    
}
void fill_data(stk::VolumeFloat3& d) {
    std::mt19937 gen(_data_seed);
    std::uniform_int_distribution<> dis(0, 10);

    for (int z = 0; z < (int)d.size().z; ++z) {
    for (int y = 0; y < (int)d.size().y; ++y) {
    for (int x = 0; x < (int)d.size().x; ++x) {
        d(x,y,z) = float3{
            (float)dis(gen) / 5.0f,
            (float)dis(gen) / 5.0f,
            (float)dis(gen) / 5.0f
        };
    }
    }
    }    
}

// Pads a float3 with a zero to get a float4 vectorfield for CUDA
static stk::VolumeFloat4 convert_vectorfield(const stk::VolumeFloat3& in)
{
    stk::VolumeFloat4 out(in.size());

    for (int z = 0; z < (int)in.size().z; ++z) {
    for (int y = 0; y < (int)in.size().y; ++y) {
    for (int x = 0; x < (int)in.size().x; ++x) {
        out(x,y,z) = {in(x,y,z).x, in(x,y,z).y, in(x,y,z).z, 0.0f};
    }
    }
    }
    return out;
}


TEST_CASE("gpu_cost_ssd", "")
{
    dim3 dims{32,32,32};

    stk::VolumeFloat fixed(dims);
    stk::VolumeFloat moving(dims);
    stk::VolumeFloat3 df(dims);
    stk::VolumeFloat cpu_cost(dims);

    fill_data(fixed);
    fill_data(moving);
    fill_data(df);
    
    SquaredDistanceFunction<float> cpu_fn(fixed, moving);
    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        cpu_cost(x,y,z) = cpu_fn.cost({x,y,z}, df(x,y,z));
    }
    }
    }

    stk::GpuVolume gpu_fixed(fixed);
    stk::GpuVolume gpu_moving(moving);
    stk::GpuVolume gpu_df(convert_vectorfield(df));
    stk::GpuVolume gpu_cost(dims, stk::Type_Float);

    gpu::run_ssd_kernel(gpu_fixed, gpu_moving, gpu_df, gpu_cost);

    stk::VolumeFloat cost_on_cpu = gpu_cost.download();

    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        REQUIRE(cost_on_cpu(x,y,z) == Approx(cpu_cost(x,y,z)).epsilon(0.001));
    }
    }
    }
}
TEST_CASE("gpu_cost_ncc", "")
{
    dim3 dims{32,32,32};

    stk::VolumeFloat fixed(dims);
    stk::VolumeFloat moving(dims);
    stk::VolumeFloat3 df(dims);
    stk::VolumeFloat cpu_cost(dims);

    fill_data(fixed);
    fill_data(moving);
    fill_data(df);
    
    NCCFunction<float> cpu_fn(fixed, moving, 2);
    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        cpu_cost(x,y,z) = cpu_fn.cost({x,y,z}, df(x,y,z));
    }
    }
    }

    stk::GpuVolume gpu_fixed(fixed);
    stk::GpuVolume gpu_moving(moving);
    stk::GpuVolume gpu_df(convert_vectorfield(df));
    stk::GpuVolume gpu_cost(dims, stk::Type_Float);

    gpu::run_ncc_kernel(gpu_fixed, gpu_moving, gpu_df, 2, gpu_cost);

    stk::VolumeFloat cost_on_cpu = gpu_cost.download();

    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        REQUIRE(cost_on_cpu(x,y,z) == Approx(cpu_cost(x,y,z)).epsilon(0.001));
    }
    }
    }
}
TEST_CASE("gpu_cost_regularizer", "")
{
    dim3 dims{32,32,32};

    stk::VolumeFloat3 df(dims);
    stk::VolumeFloat3 zero_df(dims, float3{0,0,0});
    stk::VolumeFloat3 cpu_cost(dims);

    fill_data(df);
    
    Regularizer cpu_fn(1.0f);
    cpu_fn.set_initial_displacement(zero_df);

    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        float3 r{0};

        if (x + 1 < int(dims.x)) {
            r.x = (float)cpu_fn(
                int3{x,y,z}, 
                df(x,y,z),
                df(x+1,y,z),
                {1,0,0}
            );
        }

        if (y + 1 < int(dims.y)) {
            r.y = (float)cpu_fn(
                int3{x,y,z}, 
                df(x,y,z),
                df(x,y+1,z),
                {0,1,0}
            );
        }
        
        if (z + 1 < int(dims.z)) {
            r.z = (float)cpu_fn(
                int3{x,y,z}, 
                df(x,y,z),
                df(x,y,z+1),
                {0,0,1}
            );
        }

        cpu_cost(x,y,z) = r;
    }
    }
    }

    stk::GpuVolume gpu_df(convert_vectorfield(df));
    stk::GpuVolume gpu_zero_df(convert_vectorfield(zero_df));
    stk::GpuVolume gpu_cost(dims, stk::Type_Float4);

    gpu::run_regularizer_kernel(gpu_df, gpu_zero_df, gpu_cost);

    stk::VolumeFloat4 cost_on_cpu = gpu_cost.download();

    for (int z = 0; z < (int)dims.z; ++z) {
    for (int y = 0; y < (int)dims.y; ++y) {
    for (int x = 0; x < (int)dims.x; ++x) {
        REQUIRE(cost_on_cpu(x,y,z).x == Approx(cpu_cost(x,y,z).x));
        REQUIRE(cost_on_cpu(x,y,z).y == Approx(cpu_cost(x,y,z).y));
        REQUIRE(cost_on_cpu(x,y,z).z == Approx(cpu_cost(x,y,z).z));
    }
    }
    }
}





#endif // DF_USE_CUDA
