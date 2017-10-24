#include "config.h"

#ifdef DF_ENABLE_BENCHMARK

#include "registration/blocked_graph_cut_optimizer.h"

#include <framework/job/job_system.h>
#include <framework/math/float3.h>
#include <framework/math/int3.h>
#include <framework/platform/timer.h>
#include <framework/profiler/profiler.h>

struct UnaryFn
{
    UnaryFn(VolumeFloat& fixed, VolumeFloat& moving) : 
        _fixed(fixed),
        _moving(moving)
    {}

    inline float operator()(const int3& p, const float3& def)
    {
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
struct BinaryFn
{
    BinaryFn(const float3& spacing) : _spacing(spacing) {}

    inline float operator()(const int3& /*p*/, const float3& def0, const float3& def1, const int3& step)
    {
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

    VolumeFloat3 def(Dims{100,100,100}, float3{0});

    VolumeFloat fixed(Dims{100,100,100}, 0.0f);
    VolumeFloat moving(Dims{100,100,100}, 0.0f);

    for (int z = 0; z < 100; ++z)
    {
        for (int y = 0; y < 100; ++y)
        {
            for (int x = 0; x < 100; ++x)
            {
                fixed(x, y, z) = sinf(float(x))+1.0f;
                moving(x, y, z) = cosf(float(x))+1.0f;
            }
        }
    }

    Optimizer optimizer(int3{25, 25, 25});
    
    UnaryFn unary_fn(fixed, moving);
    BinaryFn binary_fn(float3{1,1,1});

    double t_begin = timer::seconds();
    optimizer.execute(unary_fn, binary_fn, float3{1.0f, 1.0f, 1.0f}, def);
    double t_end = timer::seconds();

    printf("Elapsed: %f\n", t_end - t_begin);
}

int run_benchmark(int argc, char* argv[])
{
    argc; argv;

    job::initialize(8);

    PROFILER_REGISTER_THREAD("main");
    PROFILER_INITIALIZE();
    
    do_blocked_graph_cut_benchmark();

	MicroProfileDumpFileImmediately("benchmark.html", "benchmark.csv", NULL);

    PROFILER_SHUTDOWN();

    job::shutdown();

    return 0;
}
#endif // DF_ENABLE_BENCHMARK
