#include "config.h"

#ifdef DF_ENABLE_BENCHMARK

#include "registration/blocked_graph_cut_optimizer.h"

#include <framework/job/job_system.h>
#include <framework/math/float3.h>
#include <framework/math/int3.h>
#include <framework/platform/timer.h>
#include <framework/profiler/microprofile.h>
#include <framework/volume/volume_helper.h>
#include <framework/volume/vtk.h>

#include <omp.h>


#ifdef DF_BLOCKWISE_COST_FUNCTION
    struct UnaryFn
    {
        UnaryFn(VolumeFloat& fixed, VolumeFloat& moving) : 
            _fixed(fixed),
            _moving(moving)
        {}

        void operator()(const int3& begin, const int3& end, const VolumeFloat3& def, 
            const float3& delta, float* cost0, float* cost1)
        {
            MICROPROFILE_SCOPEI("main", "unary_fn_block", MP_MEDIUMORCHID1);

            int i = 0;
            for (int z = begin.z; z < end.z; ++z)
            {
                for (int y = begin.y; y < end.y; ++y)
                {
                    for (int x = begin.x; x < end.x; ++x)
                    {
                        if (x < 0 || x >= int(_fixed.size().width) ||
                            y < 0 || y >= int(_fixed.size().height) ||
                            z < 0 || z >= int(_fixed.size().depth))
                        {
                            continue;
                        }

                        int3 p {x, y, z};

                        float3 fixed_p0{
                            float(p.x) + def(p).x,
                            float(p.y) + def(p).y,
                            float(p.z) + def(p).z
                        };
                        float3 fixed_p1{
                            float(p.x) + def(p).x + delta.x,
                            float(p.y) + def(p).y + delta.y,
                            float(p.z) + def(p).z + delta.z
                        };

                        float3 world_p0 = _fixed.origin() + fixed_p0 * _fixed.spacing();
                        float3 world_p1 = _fixed.origin() + fixed_p1 * _fixed.spacing();
                        float3 moving_p0 = (world_p0 / _moving.spacing()) - _moving.origin();
                        float3 moving_p1 = (world_p1 / _moving.spacing()) - _moving.origin();
                
                        float moving_v0 = _moving.linear_at(moving_p0, volume::Border_Constant);
                        float moving_v1 = _moving.linear_at(moving_p1, volume::Border_Constant);

                        float f0 = fabs(float(_fixed(p) - moving_v0));
                        float f1 = fabs(float(_fixed(p) - moving_v1));

                        cost0[i] = f0 * f0;
                        cost1[i] = f1 * f1;

                        ++i;
                    }
                }
            }
        }
        VolumeFloat _fixed;
        VolumeFloat _moving;
    };

    // struct BinaryFn
    // {
    //     BinaryFn(const float3& spacing) : _spacing(spacing) {}

    //     inline float operator()(const int3& begin, const int3& end, const VolumeFloat3& def,
    //         const float3& delta, const int3& step)
    //     {
    //         //MICROPROFILE_SCOPEI("main", "binary_fn", MP_MEDIUMORCHID1);
            
    //         float3 step_in_mm {step.x*_spacing.x, step.y*_spacing.y, step.z*_spacing.z};
            
    //         float3 diff = def0 - def1;
    //         float3 diff_in_mm {diff.x*_spacing.x, diff.y*_spacing.y, diff.z*_spacing.z};
            
    //         float dist_squared = math::length_squared(diff_in_mm);
    //         float step_squared = math::length_squared(step_in_mm);
            
    //         float w = 0.1f;
    //         return w * dist_squared / step_squared;
    //     }

    //     float3 _spacing;
    // };

#else
    struct UnaryFn
    {
        UnaryFn(VolumeFloat& fixed, VolumeFloat& moving) : 
            _fixed(fixed),
            _moving(moving)
        {}

        inline float operator()(const int3& p, const float3& def)
        {
            //MICROPROFILE_SCOPEI("main", "unary_fn", MP_MEDIUMORCHID1);
            
            float3 fixed_p{
                float(p.x) + def.x,
                float(p.y) + def.y,
                float(p.z) + def.z
            }; 
            
            // [fixed] -> [world] -> [moving]
            float3 world_p = _fixed.origin() + fixed_p * _fixed.spacing();
            float3 moving_p = (world_p / _moving.spacing()) - _moving.origin();

            float moving_v = _moving.linear_at(moving_p, volume::Border_Constant);

            // TODO: Float cast
            float f = fabs(float(_fixed(p) - moving_v));
            return f*f;// PERF: f*f around 10% faster than powf(f, 2.0f);
        }

        VolumeFloat _fixed;
        VolumeFloat _moving;
    };

#endif

struct BinaryFn
{
    BinaryFn(const float3& spacing) : _spacing(spacing) {}

    inline float operator()(const int3& /*p*/, const float3& def0, const float3& def1, const int3& step)
    {
        //MICROPROFILE_SCOPEI("main", "binary_fn", MP_MEDIUMORCHID1);
        
        float3 step_in_mm {step.x*_spacing.x, step.y*_spacing.y, step.z*_spacing.z};
        
        float3 diff = def0 - def1;
        float3 diff_in_mm {diff.x*_spacing.x, diff.y*_spacing.y, diff.z*_spacing.z};
        
        float dist_squared = math::length_squared(diff_in_mm);
        float step_squared = math::length_squared(step_in_mm);
        
        float w = 0.1f;
        return w * dist_squared / step_squared;
    }

    float3 _spacing;
};

void do_blocked_graph_cut_benchmark()
{
    typedef BlockedGraphCutOptimizer<
        UnaryFn,
        BinaryFn
    > Optimizer;

    Dims img_size = {100,100,100};

    VolumeFloat3 def(img_size, float3{0});

    VolumeFloat fixed(img_size, 0.0f);
    VolumeFloat moving(img_size, 0.0f);

    for (int z = 0; z < int(img_size.depth); ++z)
    {
        for (int y = 0; y < int(img_size.height); ++y)
        {
            for (int x = 0; x < int(img_size.width); ++x)
            {
                fixed(x, y, z) = sinf(float(x))+1.0f;
                moving(x, y, z) = cosf(float(x))+1.0f;
            }
        }
    }

    Optimizer optimizer(int3{12, 12, 12});
    
    UnaryFn unary_fn(fixed, moving);
    BinaryFn binary_fn(float3{1,1,1});

    double t_begin = timer::seconds();
    optimizer.execute(unary_fn, binary_fn, float3{1.0f, 1.0f, 1.0f}, def);
    double t_end = timer::seconds();

    printf("Elapsed: %f\n", t_end - t_begin);

    vtk::write_volume("test.vtk", def);
}

int run_benchmark(int argc, char* argv[])
{
    argc; argv;

    MicroProfileOnThreadCreate("main");
    
    // Name all OpenMP threads for profiler
    auto main_thread = omp_get_thread_num();
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < 8; ++i)
    {
        if (omp_get_thread_num() != main_thread)
            MicroProfileOnThreadCreate("omp_worker");
    }

    MicroProfileSetEnableAllGroups(true);
    MicroProfileSetForceMetaCounters(true);
    //MicroProfileStartContextSwitchTrace();

    do_blocked_graph_cut_benchmark();

	MicroProfileDumpFileImmediately("benchmark.html", "benchmark.csv", NULL);

    MicroProfileShutdown();

    return 0;
}
#endif // DF_ENABLE_BENCHMARK
