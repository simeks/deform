#pragma once

#include "sub_function.h"

#include <vector>

class GpuUnaryFunction
{
public:
    struct WeightedFunction {
        float weight;
        std::unique_ptr<GpuSubFunction> function;
    };

    GpuUnaryFunction() {}
    ~GpuUnaryFunction() {}


    // cost_acc : Cost accumulator for unary term. float2 with E0 and E1.
    void operator()(
        stk::GpuVolume& df,
        const float3& delta,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cap_source,
        stk::GpuVolume& cap_sink,
        stk::cuda::Stream& stream
    )
    {
        for (auto& fn : _functions) {
            fn.function->cost(df, delta, fn.weight,
                              offset, dims, cap_source, cap_sink,
                              stream);
        }
    }

    void add_function(std::unique_ptr<GpuSubFunction>& fn, float weight)
    {
        _functions.push_back({weight, std::move(fn)});
    }

private:
    std::vector<WeightedFunction> _functions;
};

