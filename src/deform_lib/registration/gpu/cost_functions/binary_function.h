#pragma once

#include <stk/image/gpu_volume.h>

#include <deform_lib/registration/settings.h>

#include "../gpu_displacement_field.h"

class GpuBinaryFunction
{
public:
    GpuBinaryFunction() :
        _weight(0.0f),
        _scale(1.0f),
        _half_exponent(1.0f),
        _spacing{0, 0, 0}
    {
    }
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

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    void set_weight_map(const stk::GpuVolume& map) { _weight_map = map; }
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP

    // Computes the regularization cost in three directions (x+, y+, z+), with and without
    //  applied delta. Results are stored into the three provided cost volumes (of type float2)
    // df           : Displacement field
    // delta        : Delta applied to the displacement, typically based on the step-size.
    // weight       : Regularization weight
    // offset       : Offset to region to compute terms in
    // dims         : Size of region
    // cost_x       : Destination for cost in x+ direction {E00, E01, E10, E11}
    // cost_y       : Destination for cost in y+ direction {E00, E01, E10, E11}
    // cost_z       : Destination for cost in z+ direction {E00, E01, E10, E11}
    void operator()(
        const GpuDisplacementField& df,
        const float3& delta,
        const int3& offset,
        const int3& dims,
        Settings::UpdateRule update_rule,
        stk::GpuVolume& cost_x,
        stk::GpuVolume& cost_y,
        stk::GpuVolume& cost_z,
        stk::cuda::Stream& stream
    );

private:
    float _weight;
    float _scale;
    float _half_exponent;
    float3 _spacing;

#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
    stk::GpuVolume _weight_map;
#endif // DF_ENABLE_REGULARIZATION_WEIGHT_MAP
};

