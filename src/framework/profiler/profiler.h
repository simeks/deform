#if 0

#pragma once

#define DF_USE_PROFILING 0

#ifdef DF_USE_PROFILING
#include "microprofile.h"
#endif

namespace profiler
{
#ifdef DF_USE_PROFILING
    /// Initializes the profiler
    void initialize();

    /// Shuts down profilinng
    void shutdown();

    /// Registers a thread to the profiler, 
    ///		this should be done for all newly created threads
    void register_thread(const char* name);

    /// Marks a frame, should be called once every frame
    void frame_tick();

#else
    void initialize() {}
    void shutdown() {}
    void register_thread(const char*) {}
    void frame_tick() {}
#endif

}
#endif
