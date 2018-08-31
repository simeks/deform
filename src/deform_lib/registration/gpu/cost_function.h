#pragma once

#include <stk/common/assert.h>
#include <stk/image/dim3.h>
#include <stk/image/gpu_volume.h>

#include <memory>
#include <tuple>
#include <vector>

namespace gpu {
    // Computes the regularization cost in three directions (x+, y+, z+), with and without
    //  applied delta. Results are stored into the three provided cost volumes (of type float2)
    // df           : Displacement field
    // initial_df   : Initial displacement field of current level
    // cost_x       : Destination for cost in x+ direction (before and after applied delta)
    // cost_y       : Destination for cost in y+ direction (before and after applied delta)
    // cost_z       : Destination for cost in z+ direction (before and after applied delta)
    // delta    : Delta applied to the displacement, typically based on the step-size.
    void run_regularizer_kernel(
        const stk::GpuVolume& df,
        const stk::GpuVolume& initial_df,
        stk::GpuVolume& cost_x, // float2
        stk::GpuVolume& cost_y, // float2
        stk::GpuVolume& cost_z, // float2
        float3 delta,
        const dim3& block_size = {32,32,1}
    );

    // Computes the SSD with and without applied displacement delta. Costs are accumulated
    //  into the specified cost_acc volume (of type float2), where x is the cost before applying 
    //  delta and y is the cost after.
    // df       : Displacement field
    // cost_acc : Destination for cost (float2, with cost before (x) and after (y) applying delta)
    // delta    : Delta applied to the displacement, typically based on the step-size.
    void run_ssd_kernel(
        const stk::GpuVolume& fixed,
        const stk::GpuVolume& moving,
        const stk::GpuVolume& df,
        stk::GpuVolume& cost_acc, // float2
        float3 delta,
        const dim3& block_size = {32,32,1}
    );
    
    // Computes the NCC with and without applied displacement delta. Costs are accumulated
    //  into the specified cost_acc volume (of type float2), where x is the cost before applying 
    //  delta and y is the cost after.
    // df       : Displacement field
    // cost_acc : Destination for cost (float2, with cost before (x) and after (y) applying delta)
    // delta    : Delta applied to the displacement, typically based on the step-size.
    void run_ncc_kernel(
        const stk::GpuVolume& fixed,
        const stk::GpuVolume& moving,
        const stk::GpuVolume& df,
        int radius,
        float3 delta,
        stk::GpuVolume& cost_acc, // float2
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
            fn.function->cost(df, cost_acc);
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

struct GpuBinaryFunction
{
    GpuBinaryFunction() {}
    ~GpuBinaryFunction() {}

    // Sets the initial displacement for this registration level. This will be
    //  the reference when computing the regularization energy. Any displacement 
    //  identical to the initial displacement will result in zero energy.
    void set_initial_displacement(const stk::GpuVolume& initial)
    {
        ASSERT(initial.voxel_type() == stk::Type_Float4);
        _initial = initial;
    }

    void operator()(const stk::GpuVolume& df, stk::GpuVolume& cost_acc)
    {
        gpu::run_regularizer_kernel(df, _initial, cost_acc);
    }

    stk::GpuVolume _initial;
};
