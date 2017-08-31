#pragma once

template<typename TImage>
class BlockedGraphCutOptimizer : public Optimizer
{
public:
    BlockedGraphCutOptimizer(const Dict& settings);
    ~BlockedGraphCutOptimizer();

    void execute(
        const Image* fixed, 
        const Image* moving, 
        int pair_count,
        ImageVec3d& def) override;

private:
    bool do_block(
        const Vec3i& block_p, 
        const Vec3i& block_dims, 
        const Vec3i& block_offset, 
        const Vec3d& delta,
        const ImageUInt8& constraint_mask,
        ImageVec3d& def);

    double calculate_energy(ImageVec3d& def, const Vec3i& dims);

    Vec3i _neighbors[6];
    Vec3i _block_size;
    double _step_size;

    EnergyFunction<TImage> _energy;
};

#include "BlockedGraphCutOptimizer.inl"
