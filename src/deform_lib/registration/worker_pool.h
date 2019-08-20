#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class WorkerPool
{
public:
    WorkerPool(int n_threads) : _stop(false)
    {
        for (int i = 0; i < n_threads; ++i) {
            _workers.emplace_back([&](){
                while (!_stop) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(_queue_lock);
                        _cv.wait(lock, [this] {
                            return this->_stop || !this->_queue.empty();
                        });

                        if(_stop && _queue.empty())
                            return;

                        task = std::move(_queue.front());
                        _queue.pop_front();
                    }
                    task();
                }
            });
        }
    }
    ~WorkerPool()
    {
        _stop = true;

        _cv.notify_all();
        for (auto& w : _workers) {
            w.join();
        }
    }

    // Push high-priority task to top of queue
    void push_front(std::function<void()> fn)
    {
        std::unique_lock<std::mutex> lock(_queue_lock);
        _queue.push_front(fn);

        _cv.notify_one();
    }

    // Push task to end of queue
    void push_back(std::function<void()> fn)
    {
        std::unique_lock<std::mutex> lock(_queue_lock);
        _queue.push_back(fn);

        _cv.notify_one();
    }

    // Attempts to pop a task from the queue
    // returns nullptr if no task was available
    std::function<void()> try_pop()
    {
        std::unique_lock<std::mutex> lock(_queue_lock);
        if (_queue.empty())
            return nullptr;

        auto task = std::move(_queue.front());
        _queue.pop_front();
        return task;
    }

private:
    std::vector<std::thread> _workers;

    std::deque<std::function<void()>> _queue;
    std::mutex _queue_lock;

    std::condition_variable _cv;

    std::atomic<bool> _stop;
};
