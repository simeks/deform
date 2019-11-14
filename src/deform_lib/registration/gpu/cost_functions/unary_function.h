#pragma once

#include <deform_lib/registration/settings.h>

#include "cost_function.h"

#include <vector>

class GpuDisplacementField;

// A set of weighted cost functions
class GpuUnaryFunction
{
public:
    struct WeightedFunction {
        float weight;
        std::unique_ptr<GpuCostFunction> function;
    };

    GpuUnaryFunction() {}
    ~GpuUnaryFunction() {}

    // cost_acc : Cost accumulator for unary term. float2 with E0 and E1.
    void operator()(
        GpuDisplacementField& df,
        const float3& delta,
        const int3& offset,
        const int3& dims,
        Settings::UpdateRule update_rule,
        stk::GpuVolume& cost_acc,
        stk::cuda::Stream& stream
    )
    {
        for (auto& fn : _functions) {
            fn.function->cost(
                df,
                delta,
                fn.weight,
                offset,
                dims,
                update_rule,
                cost_acc,
                stream
            );
        }
    }

    void add_function(std::unique_ptr<GpuCostFunction>& fn, float weight)
    {
        _functions.push_back({weight, std::move(fn)});
    }

private:
    std::vector<WeightedFunction> _functions;
};

