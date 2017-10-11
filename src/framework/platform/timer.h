#pragma once

#include <stdint.h>

namespace timer
{
    void initialize();

    /// Returns the number of ticks since timer initialization.
    uint64_t start_tick_count();

    /// Returns the applications current tick count.
    uint64_t tick_count();

    /// Returns elapsed seconds since timer initialization.
    double seconds();

    /// Returns the number of seconds per timer tick.
    double seconds_per_tick();
    
}
