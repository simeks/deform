#pragma once

#ifndef DF_PLATFORM_LINUX
    #define DF_USE_PROFILING
#endif

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
#endif

}

#ifdef DF_USE_PROFILING
    #define PROFILER_INITIALIZE() profiler::initialize()
    #define PROFILER_SHUTDOWN() profiler::shutdown()
    #define PROFILER_REGISTER_THREAD(name) profiler::register_thread(name)
    #define PROFILER_FRAME_TICK() profiler::frame_tick()
#else
    #define PROFILER_INITIALIZE()
    #define PROFILER_SHUTDOWN()
    #define PROFILER_REGISTER_THREAD(name)
    #define PROFILER_FRAME_TICK()
#endif
