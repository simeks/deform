#pragma once

#if defined(DF_ENABLE_MICROPROFILE)

#include "microprofile.h"

#define PROFILER_PRINT_FILE "profiler.html"

#define PROFILER_INIT() \
    MicroProfileSetEnableAllGroups(true); \
    MicroProfileSetForceMetaCounters(true);
#define PROFILER_SHUTDOWN() \
	MicroProfileDumpFileImmediately("profiler.html", "profiler.csv", 0); \
    MicroProfileShutdown();
#define PROFILER_REGISTER_THREAD(name)

#define PROFILER_FLIP() MicroProfileFlip(0)

#define PROFILER_SCOPE(name, color) MICROPROFILE_SCOPEI("deform", name, color)
#define PROFILER_COUNTER_SET(name, v) MICROPROFILE_COUNTER_SET(name, v)

#elif defined(DF_ENABLE_NVTOOLSEXT)

#include <nvToolsExt.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <windows.h>
#endif

#define PROFILER_INIT()
#define PROFILER_SHUTDOWN()
#define PROFILER_REGISTER_THREAD(name) stk::cuda::register_current_thread(name)

#define PROFILER_FLIP()

#define PROFILER_SCOPE_VAR2(line) _profiler_##line
#define PROFILER_SCOPE_VAR(line) PROFILER_SCOPE_VAR2(line)
#define PROFILER_SCOPE(name, color) stk::cuda::ProfilerScope PROFILER_SCOPE_VAR(__LINE__)(name, color)
#define PROFILER_COUNTER_SET(name, v)

namespace stk {
namespace cuda {

    struct ProfilerScope
    {
        ProfilerScope(const char* name, uint32_t color) {
            nvtxEventAttributes_t attr = {0};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.colorType = NVTX_COLOR_ARGB;
            attr.color = color;
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = name;
            nvtxRangePushEx(&attr);
        }
        ~ProfilerScope() {
            nvtxRangePop();
        }
    };

    inline void register_thread(const char* name) {
        nvtxNameOsThread(
        #ifdef _WIN32
            ::GetCurrentThreadId(), 
        #else
            pthread_self(),
        #endif
            name);
    }
}
}

#else


#define PROFILER_INIT()
#define PROFILER_SHUTDOWN()
#define PROFILER_REGISTER_THREAD(name)

#define PROFILER_FLIP()

#define PROFILER_SCOPE(name, color)
#define PROFILER_COUNTER_SET(name, v)

#endif
