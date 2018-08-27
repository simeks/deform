#pragma once

#include <stk/image/dim3.h>
#include <stk/image/gpu_volume.h>

#include <memory>
#include <tuple>
#include <vector>

namespace gpu {
    // df           : Displacement field
    // initial_df   : Initial displacement field of current level
    // cost         : Destination for cost (float4, with cost in x+, y+, z+)
    void run_regularizer_kernel(
        const stk::GpuVolume& df,
        const stk::GpuVolume& initial_df,
        stk::GpuVolume& cost,
        const dim3& block_size = {32,32,1}
    );
    void run_ssd_kernel(
        const stk::GpuVolume& fixed,
        const stk::GpuVolume& moving,
        const stk::GpuVolume& df,
        stk::GpuVolume& cost_acc,
        const dim3& block_size = {32,32,1}
    );
    void run_ncc_kernel(
        const stk::GpuVolume& fixed,
        const stk::GpuVolume& moving,
        const stk::GpuVolume& df,
        int radius,
        stk::GpuVolume& cost_acc,
        const dim3& block_size = {32,32,1}
    );
}

struct GpuSubFunction
{
    /// cost_acc : Cost accumulator
    virtual void cost(const stk::GpuVolume& df, stk::GpuVolume& cost_acc) = 0;
};

struct GpuCostFunction_SSD : public GpuSubFunction
{
    GpuCostFunction_SSD(const stk::GpuVolume& fixed, 
                        const stk::GpuVolume& moving) :
        _fixed(fixed),
        _moving(moving)
    {
    }
    ~GpuCostFunction_SSD() {}

    void cost(const stk::GpuVolume& df, stk::GpuVolume& cost_acc)
    {
        gpu::run_ssd_kernel(_fixed, _moving, df, cost_acc);
    }

    stk::GpuVolume _fixed;
    stk::GpuVolume _moving;
};


struct GpuCostFunction_NCC : public GpuSubFunction
{
    GpuCostFunction_NCC(const stk::GpuVolume& fixed, 
                        const stk::GpuVolume& moving,
                        int radius) :
        _fixed(fixed), 
        _moving(moving),
        _radius(radius)
    {
    }
    ~GpuCostFunction_NCC() {}

    void cost(const stk::GpuVolume& df, stk::GpuVolume& cost_acc)
    {
        gpu::run_ncc_kernel(_fixed, _moving, df, _radius, cost_acc);
    }

    stk::GpuVolume _fixed;
    stk::GpuVolume _moving;
    int _radius;
};

struct GpuUnaryFunction
{
    struct WeightedFunction {
        float weight;
        std::unique_ptr<GpuSubFunction> function;
    };

    // TODO: Weights, regularization term, etc

    GpuUnaryFunction() {}
    ~GpuUnaryFunction() {}

    void operator()(const stk::GpuVolume& df, stk::GpuVolume& cost_acc)
    {
        for (auto& fn : _functions) {
            fn->cost(df, cost_acc);
        }
        // TODO: Maybe applying regularization as a separate pass?
        //       Would make sense for regularization weight maps.
    }

    void add_function(std::unique_ptr<GpuSubFunction> fn, float weight)
    {
        _functions.push_back({weight, std::move(fn)});
    }

    std::vector<WeightedFunction> _functions;
};
