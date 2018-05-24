#pragma once

#include <stdint.h>

namespace timer
{
    void initialize();

    /// Returns elapsed seconds since timer initialization.
    double seconds();

    
}
