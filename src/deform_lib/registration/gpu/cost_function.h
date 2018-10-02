#pragma once

#include <stk/common/assert.h>
#include <stk/image/dim3.h>
#include <stk/image/gpu_volume.h>

#include <memory>
#include <tuple>
#include <vector>

namespace stk { namespace cuda {
    class Stream;
}}

struct GpuSubFunction
{
    // Costs are accumulated into the specified cost_acc volume (of type float2), 
    //  where x is the cost before applying delta and y is the cost after.
    // offset       : Offset to region to compute terms in
    // dims         : Size of region
    // df           : Displacement field
    // cost_acc     : Destination for cost (float2, with cost before (x) and after (y) applying delta)
    // delta        : Delta applied to the displacement, typically based on the step-size.
    virtual void cost(
        stk::GpuVolume& df,
        const float3& delta,
        float weight, 
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cost_acc,
        stk::cuda::Stream& stream
    ) = 0;
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

    void cost(
        stk::GpuVolume& df,
        const float3& delta,
        float weight, 
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cost_acc,
        stk::cuda::Stream& stream
    );

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

    void cost(
        stk::GpuVolume& df,
        const float3& delta,
        float weight, 
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cost_acc,
        stk::cuda::Stream& stream
    );

    stk::GpuVolume _fixed;
    stk::GpuVolume _moving;
    int _radius;
};

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

class GpuBinaryFunction
{
public:
    GpuBinaryFunction() : _weight(0.0f), _spacing{0} {}
    ~GpuBinaryFunction() {}

    void set_regularization_weight(float weight)
    {
        _weight = weight;
    }
    void set_fixed_spacing(const float3& spacing)
    {
        _spacing = spacing;
    }

    // Sets the initial displacement for this registration level. This will be
    //  the reference when computing the regularization energy. Any displacement 
    //  identical to the initial displacement will result in zero energy.
    void set_initial_displacement(const stk::GpuVolume& initial)
    {
        ASSERT(initial.voxel_type() == stk::Type_Float4);
        ASSERT(initial.usage() == stk::gpu::Usage_PitchedPointer);

        _initial = initial;
    }

    // Computes the regularization cost in three directions (x+, y+, z+), with and without
    //  applied delta. Results are stored into the three provided cost volumes (of type float2)
    // df           : Displacement field
    // initial_df   : Initial displacement field of current level
    // delta        : Delta applied to the displacement, typically based on the step-size.
    // weight       : Regularization weight
    // offset       : Offset to region to compute terms in
    // dims         : Size of region
    // cost_x       : Destination for cost in x+ direction {E00, E01, E10, E11}
    // cost_y       : Destination for cost in y+ direction {E00, E01, E10, E11}
    // cost_z       : Destination for cost in z+ direction {E00, E01, E10, E11}
    void operator()(
        const stk::GpuVolume& df,
        const float3& delta,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cost_x,
        stk::GpuVolume& cost_y,
        stk::GpuVolume& cost_z,
        stk::cuda::Stream& stream
    );

private:
    float _weight;
    float3 _spacing;

    stk::GpuVolume _initial;
};
