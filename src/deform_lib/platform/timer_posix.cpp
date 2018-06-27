#include "assert.h"
#include "windows_wrapper.h"
#include "timer.h"

#include <time.h>

namespace
{
    double g_start_time_s = 0;

    bool g_initialized = false;
};

void timer::initialize()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);

    g_start_time_s = t.tv_sec + t.tv_nsec * double(1e-9);
    g_initialized = true;
}

double timer::seconds()
{
    assert(g_initialized);

    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    double now = t.tv_sec + t.tv_nsec * double(1e-9);
    return now - g_start_time_s;
}
