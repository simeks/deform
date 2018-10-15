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

    // TODO: Weights, regularization term, etc

    GpuUnaryFunction() : _regularization_weight(0.0f) {}

    ~GpuUnaryFunction() {}

    void set_regularization_weight(float weight)
    {
        _regularization_weight = weight;
    }

    // cost_acc : Cost accumulator for unary term. float2 with E0 and E1.
    void operator()(
        stk::GpuVolume& df,
        const float3& delta,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cost_acc,
        stk::cuda::Stream& stream
    )
    {
        for (auto& fn : _functions) {
            fn.function->cost(df, delta, (1.0f-_regularization_weight)*fn.weight,
                              offset, dims, cost_acc, stream);
        }
        // TODO: Maybe applying regularization as a separate pass?
        //       Would make sense for regularization weight maps.
    }

    void add_function(std::unique_ptr<GpuSubFunction> fn, float weight)
    {
        _functions.push_back({weight, std::move(fn)});
    }

private:
    float _regularization_weight;

    std::vector<WeightedFunction> _functions;
};

