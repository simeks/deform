#pragma once

#include <stdint.h>

namespace std
{
    template<typename>
    struct atomic;
}

namespace job
{
    typedef void(*WorkFunction)(void*);

    struct JobDecl
    {
        JobDecl(WorkFunction fn = nullptr, void* data = nullptr) :
            fn(fn), data(data)
        {
        }

        WorkFunction fn;
        void* data;
    };

    void initialize(int num_workers);
    void shutdown();

    void run_jobs(JobDecl* jobs, int num_job, std::atomic<int>& counter);
    void wait_for(std::atomic<int>& counter);
}

