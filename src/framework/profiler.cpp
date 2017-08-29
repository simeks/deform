#include "profiler.h"

#ifdef DF_ENABLE_CUDA
#include <nvToolsExt.h>
#include <windows.h>

ProfileScope::ProfileScope(const char* name)
{
    nvtxRangePush(name);
}
ProfileScope::~ProfileScope()
{
    nvtxRangePop();
}

namespace profiler
{
    void profiler::register_current_thread(const char* name)
    {
        nvtxNameOsThread(::GetCurrentThreadId(), name);
    }
}
#endif // DF_ENABLE_CUDA