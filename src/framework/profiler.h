#pragma once


#ifdef DF_ENABLE_CUDA
struct ProfileScope
{
    ProfileScope(const char* name);
    ~ProfileScope();
};

namespace profiler 
{
    void register_current_thread(const char* name);
}

#define PROFILE_SCOPE_VAR(line) _profiler_##line
#define PROFILE_SCOPE(name) ProfileScope PROFILE_SCOPE_VAR(__LINE__)(name)
#else
// Profiler not enabled when CUDA is disabled
#define PROFILE_SCOPE_VAR(line)
#define PROFILE_SCOPE(name)
#endif

