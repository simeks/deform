#include "debug/assert.h"
#include "job_system.h"
#include "profiler/microprofile.h"
#include "thread/lock.h"
#include "thread/thread.h"

#include <atomic>
#include <deque>
#include <iostream>
#include <vector>

using thread::CriticalSection;
using thread::ScopedLock;
using thread::Thread;

#define YIELD() YieldProcessor()

namespace job
{
    struct JobItem
    {
        WorkFunction fn;
        void* data;

        std::atomic<int>* counter;
    };

    class JobSystem;
    class Worker
    {
    public:
        Worker(JobSystem* owner);
        ~Worker();

        /// Attempts to process one job item from the queue
        void do_work();

    private:
        static void worker_proc(Worker* worker);

        JobSystem* _owner;
        thread::Thread _thread;
    };

    class JobSystem
    {
    public:
        JobSystem(int num_workers);
        ~JobSystem();

        bool is_shutting_down() const;

        void push_job(const JobItem& job);
        bool pop_job(JobItem& out);

    private:
        std::vector<Worker*> _workers;

        bool _shutting_down;

        std::deque<JobItem> _queue;
        mutable thread::CriticalSection _queue_lock; // Mutable to enable thread-safe const methods
    };

    namespace internal
    {
        JobSystem* _job_system = nullptr;
        thread_local Worker* _thread_local_worker;
    }

    Worker::Worker(JobSystem* owner) : _owner(owner)
    {
        _thread.start((thread::Thread::Function)worker_proc, this);
    }
    Worker::~Worker()
    {
        _thread.join();
    }
    void Worker::do_work()
    {
        JobItem item;
        if (!_owner->pop_job(item))
            YIELD();
        else
        {
            MICROPROFILE_SCOPEI("main", "job", 0xFF1132FF);

            assert(item.fn);
            item.fn(item.data);

            --(*item.counter);
        }
    }

    void Worker::worker_proc(Worker* worker)
    {
        internal::_thread_local_worker = worker;

        while (!worker->_owner->is_shutting_down())
        {
            worker->do_work();
        }
    }

    JobSystem::JobSystem(int num_workers) :
        _shutting_down(false)
    {
        for (int i = 0; i < num_workers; ++i)
        {
            _workers.push_back(new Worker(this));
        }
    }
    JobSystem::~JobSystem()
    {
        _shutting_down = true;

        for (auto worker : _workers)
        {
            // Wait for workers to stop
            delete worker;
        }
        _workers.clear();
    }
    bool JobSystem::is_shutting_down() const
    {
        return _shutting_down;
    }
    void JobSystem::push_job(const JobItem& job)
    {
        ScopedLock<CriticalSection> lock(_queue_lock);
        _queue.push_back(job);
    }
    bool JobSystem::pop_job(JobItem& out)
    {
        ScopedLock<CriticalSection> lock(_queue_lock);
        if (_queue.empty())
            return false;

        out = _queue.front();
        _queue.pop_front();
        return true;
    }

    void initialize(int num_workers)
    {
        assert(!internal::_job_system);
        internal::_job_system = new JobSystem(num_workers);
    }
    void shutdown()
    {
        assert(internal::_job_system);
        delete internal::_job_system;
        internal::_job_system = nullptr;
    }
    void job::run_jobs(JobDecl* jobs, int num_jobs, std::atomic<int>& counter)
    {
        counter += num_jobs;

        for (int i = 0; i < num_jobs; ++i)
        {
            JobItem item = { jobs[i].fn, jobs[i].data, &counter };
            internal::_job_system->push_job(item);
        }
    }
    void job::wait_for(std::atomic<int>& counter)
    {
        while (counter > 0)
        {
            // Do some work if we are a worker thread
            if (internal::_thread_local_worker)
                internal::_thread_local_worker->do_work();
            else
                YIELD();
        }
    }

} // namespace job
