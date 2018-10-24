#pragma once

#include "sub_function.h"


template<bool use_mask>
struct UnaryFunction
{
    struct WeightedFunction {
        float weight;
        std::unique_ptr<SubFunction> function;
    };

    UnaryFunction(
            const stk::VolumeFloat fixed_mask = stk::VolumeFloat(),
            const float regularization_weight = 0.0f
            )
        : _fixed_mask (fixed_mask)
        , _regularization_weight(regularization_weight)
    {
    }

    void set_fixed_mask(const stk::VolumeFloat& mask)
    {
        _fixed_mask = mask;
    }

    void set_regularization_weight(float weight)
    {
        _regularization_weight = weight;
    }
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_regularization_weight_map(stk::VolumeFloat& map)
    {
        _regularization_weight_map = map;
    }
#endif

    void add_function(std::unique_ptr<SubFunction> fn, float weight)
    {
        _functions.push_back({weight, std::move(fn)});
    }

    inline double operator()(const int3& p, const float3& def)
    {
        if constexpr (use_mask) {
            if (_fixed_mask(p) <= std::numeric_limits<float>::epsilon()) {
                return 0.0f;
            }
        }

        double sum = 0.0f;
        for (auto& fn : _functions) {
            sum += fn.weight * fn.function->cost(p, def);
        }

        float w = _regularization_weight;
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        if (_regularization_weight_map.valid())
            w = _regularization_weight_map(p);
#endif

        if constexpr (use_mask) {
            sum *= _fixed_mask(p);
        }

        return (1.0f - w) * sum;
    }

    void pre_iteration_hook(const int iteration, const stk::VolumeFloat3& def) {
        for (auto& fn : _functions) {
            fn.function->pre_iteration_hook(iteration, def);
        }
    }

    stk::VolumeFloat _fixed_mask;

    float _regularization_weight;
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    stk::VolumeFloat _regularization_weight_map;
#endif

    std::vector<WeightedFunction> _functions;
};

