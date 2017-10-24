#pragma once

#include <stdint.h>

namespace thread
{
    class Thread
    {
    public:
        typedef void(*Function)(void*);

        struct Payload
        {
            Function function;
            void* data;
        };

        Thread();
        /// Constructs and executes a thread with the given payload
        Thread(Function fn, void* data);
        ~Thread();

        /// Starts a thread (if it hasn't already been started)
        void start(Function fn, void* data);

        /// Joins with the thread
        void join();

    private:
        Thread(const Thread&);
        void operator=(const Thread&);

        void* _handle;

        Payload _payload;
    };

    

    long interlocked_increment(long volatile* addend);
    long interlocked_decrement(long volatile* addend);

    /// @return The initial value of the addend parameter.
    long interlocked_exchange_add(long volatile* addend, long value);
    unsigned long interlocked_exchange_subtract(unsigned long volatile* addend, unsigned long value);

    /// @return The initial value of the dest parameter.
    long interlocked_compare_exchange(long volatile* dest, long exchange, long comparand);

    /// @return Initial value of dest.
    long interlocked_exchange(long volatile* dest, long value);

    int64_t interlocked_increment_64(int64_t volatile* addend);
    int64_t interlocked_decrement_64(int64_t volatile* addend);

    int64_t interlocked_exchange_add_64(int64_t volatile* addend, int64_t value);

}; // namespace thread


