#pragma once

#include <stdint.h>

namespace thread
{
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


