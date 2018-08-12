#include <deform_lib/arg_parser.h>
#include <deform_lib/cost_function.h>
#include <deform_lib/filters/resample.h>

#include <stk/common/assert.h>
#include <stk/common/log.h>
#include <stk/filters/normalize.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>

#include <chrono>
#include <iostream>

using namespace std::chrono;

namespace {
    struct ScopeTimer
    {
        ScopeTimer(const char* name)
        {
            start = high_resolution_clock::now();
        }
        ~ScopeTimer()
        {
            auto stop = high_resolution_clock::now();
            double elapsed = duration_cast<std::chrono::milliseconds>(stop - start).count();
            std::cout << "Elapsed: " << elapsed << std::endl;
        }

        time_point<high_resolution_clock> start;
    };
}

int run_benchmark(int argc, char* argv[])
{
    ScopeTimer timer("asd");
    #ifdef DF_USE_CUDA

    #endif

    return 0;
}
