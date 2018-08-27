#pragma once

#include <stk/image/volume.h>
#include <stk/math/float3.h>
#include <stk/math/int3.h>
#include <stk/math/types.h>

#include "../config.h"

struct GpuUnaryFunction;
struct GpuBinaryFunction;

class GpuBlockedGraphCutOptimizer
{
public:
    GpuBlockedGraphCutOptimizer(const int3& block_size, double block_energy_epsilon);
    ~GpuBlockedGraphCutOptimizer();

    /// step_size : Step size in [mm]
    void execute(
        GpuUnaryFunction& unary_fn,
        GpuBinaryFunction& binary_fn,
        float3 step_size,
        stk::GpuVolume& df);

private:
    int3 _block_size;
    double _block_energy_epsilon;
};

