#pragma once

#include "sub_function.h"

class GpuBinaryFunction
{
public:
    GpuBinaryFunction() : _weight(0.0f), _scale(1.0f), _half_exponent(1.0f), _spacing{0, 0, 0} {}
    ~GpuBinaryFunction() {}

    void set_regularization_weight(const float weight)
    {
        _weight = weight;
    }
    void set_regularization_scale(const float scale)
    {
        _scale = scale;
    }
    void set_regularization_exponent(const float exponent)
    {
        _half_exponent = 0.5f * exponent;
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
    // cap_source   : Destination for source capacities
    // cap_sink     : Destination for sink capacities
    // cap_lee      : Destination for edge capacities [-1,  0,  0]
    // cap_gee      : Destination for edge capacities [ 1,  0,  0]
    // cap_ele      : Destination for edge capacities [ 0, -1,  0]
    // cap_ege      : Destination for edge capacities [ 0,  1,  0]
    // cap_eel      : Destination for edge capacities [ 0,  0, -1]
    // cap_eeg      : Destination for edge capacities [ 0,  0,  1]
    void operator()(
        const stk::GpuVolume& df,
        const float3& delta,
        const int3& offset,
        const int3& dims,
        stk::GpuVolume& cap_source,
        stk::GpuVolume& cap_sink,
        stk::GpuVolume& cap_lee,
        stk::GpuVolume& cap_gee,
        stk::GpuVolume& cap_ele,
        stk::GpuVolume& cap_ege,
        stk::GpuVolume& cap_eel,
        stk::GpuVolume& cap_eeg,
        stk::cuda::Stream& stream
    );

private:
    float _weight;
    float _scale;
    float _half_exponent;
    float3 _spacing;

    stk::GpuVolume _initial;
};

