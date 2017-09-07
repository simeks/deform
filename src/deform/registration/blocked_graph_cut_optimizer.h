#pragma once

const float default_regularization_weight = 0.1f;

struct Regularizer
{
    inline float operator()(const float3& def0, const float3& def1, const float3& step)
    {
        // def0, def1 : Deformation vectors are in mm
        // step : Step in mm

        float3 diff = def0 - def1;
        float dist_squared = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        float step_squared = step.x * step.x + step.y * step.y + step.z * step.z;
        return default_regularization_weight * dist_squared / step_squared;
    }
};

struct EnergyFunction
{
    inline float operator()()
    {
        return 0.0f;
    }
};

template<
    typename TRegularizer,
    typename TEnergyFunction
    >
class BlockedGraphCutOptimizer : public Optimizer
{
public:
    BlockedGraphCutOptimizer();
    ~BlockedGraphCutOptimizer();

    void execute(VolumeFloat3& def) override;

private:
    bool do_block(
        const int3& block_p, 
        const int3& block_dims, 
        const int3& block_offset, 
        const float3& delta, // delta in mm
        VolumeFloat3& def);

    int3 _neighbors[6];
    int3 _block_size;
    float _step_size;

    TEnergyFunction _unary_function;
    TRegularizer _binary_function;
};

#include "blocked_graph_cut_optimizer.inl"
