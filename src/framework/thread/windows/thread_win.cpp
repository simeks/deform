#include "debug/assert.h"
#include "platform/windows_wrapper.h"

#include "../thread.h"

#include <process.h>

static void __cdecl thread_proc(void* param)
{
    auto payload = (thread::Thread::Payload*)param;
    payload->function(payload->data);

    _endthread();
}

namespace thread
{
    Thread::Thread() :
        _handle(nullptr)
    {
        _payload.function = nullptr;
        _payload.data = nullptr;
    }
    Thread::Thread(Function fn, void* data) :
        _handle(nullptr)
    {
        _payload.function = fn;
        _payload.data = data;

        _handle = (void*)_beginthread(thread_proc, 0, &_payload);
        assert(_handle != (void*)-1L);
    }
    Thread::~Thread()
    {
    }
    void Thread::start(Function fn, void* data)
    {
        assert(_handle == nullptr || _handle == (void*)-1L);

        _payload.function = fn;
        _payload.data = data;

        _handle = (void*)_beginthread(thread_proc, 0, &_payload);
        assert(_handle != (void*)-1L);
    }
    void Thread::join()
    {
        if (_handle != nullptr)
        {
            WaitForSingleObject((HANDLE)_handle, INFINITE);
        }
    }

    long interlocked_increment(long volatile* addend)
    {
        return ::InterlockedIncrement(addend);
    }
    long interlocked_decrement(long volatile* addend)
    {
        return ::InterlockedDecrement(addend);
    }

    long interlocked_exchange_add(long volatile* addend, long value)
    {
        return ::InterlockedExchangeAdd(addend, value);
    }
    unsigned long interlocked_exchange_subtract(unsigned long volatile* addend, unsigned long value)
    {
        return ::InterlockedExchangeSubtract(addend, value);
    }
    long interlocked_compare_exchange(long volatile* dest, long exchange, long comparand)
    {
        return (::InterlockedCompareExchange(dest, exchange, comparand) == comparand);
    }
    long interlocked_exchange(long volatile* dest, long value)
    {
        return ::InterlockedExchange(dest, value);
    }

    int64_t interlocked_increment_64(int64_t volatile* addend)
    {
        return ::InterlockedIncrement64(addend);
    }
    int64_t interlocked_decrement_64(int64_t volatile* addend)
    {
        return ::InterlockedDecrement64(addend);
    }

    int64_t interlocked_exchange_add_64(int64_t volatile* addend, int64_t value)
    {
        return ::InterlockedExchangeAdd64(addend, value);
    }
}



