#pragma once

#include "sub_function.h"


struct UnaryFunction
{
    struct WeightedFunction {
        float weight;
        std::unique_ptr<SubFunction> function;
    };

    UnaryFunction() {}
    ~UnaryFunction() {}

    void set_fixed_mask(const stk::VolumeFloat& mask)
    {
        _fixed_mask = mask;
    }

    void add_function(std::unique_ptr<SubFunction> fn, float weight)
    {
        _functions.push_back({weight, std::move(fn)});
    }

    inline double operator()(const int3& p, const float3& def)
    {
        float mask_value = 1.0f;
        if (_fixed_mask.valid()) {
            mask_value = _fixed_mask(p);
            if (mask_value <= std::numeric_limits<float>::epsilon()) {
                return 0.0f;
            }
        }

        double sum = 0.0f;
        for (auto& fn : _functions) {
            sum += fn.weight * fn.function->cost(p, def);
        }

        return mask_value * sum;
    }

    void pre_iteration_hook(const int iteration, const stk::VolumeFloat3& def)
    {
        for (auto& fn : _functions) {
            fn.function->pre_iteration_hook(iteration, def);
        }
    }

    stk::VolumeFloat _fixed_mask;

    std::vector<WeightedFunction> _functions;
};

