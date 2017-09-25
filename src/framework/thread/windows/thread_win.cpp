#include "platform/windows_wrapper.h"

#include "../thread.h"

#include <process.h>


long thread::interlocked_increment(long volatile* addend)
{
    return ::InterlockedIncrement(addend);
}
long thread::interlocked_decrement(long volatile* addend)
{
    return ::InterlockedDecrement(addend);
}

long thread::interlocked_exchange_add(long volatile* addend, long value)
{
    return ::InterlockedExchangeAdd(addend, value);
}
unsigned long thread::interlocked_exchange_subtract(unsigned long volatile* addend, unsigned long value)
{
    return ::InterlockedExchangeSubtract(addend, value);
}
long thread::interlocked_compare_exchange(long volatile* dest, long exchange, long comparand)
{
    return (::InterlockedCompareExchange(dest, exchange, comparand) == comparand);
}
long thread::interlocked_exchange(long volatile* dest, long value)
{
    return ::InterlockedExchange(dest, value);
}

int64_t thread::interlocked_increment_64(int64_t volatile* addend)
{
    return ::InterlockedIncrement64(addend);
}
int64_t thread::interlocked_decrement_64(int64_t volatile* addend)
{
    return ::InterlockedDecrement64(addend);
}

int64_t thread::interlocked_exchange_add_64(int64_t volatile* addend, int64_t value)
{
    return ::InterlockedExchangeAdd64(addend, value);
}



