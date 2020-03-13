#pragma once

class GpuDisplacementField;
class GpuVolumePyramid;

namespace stk {
    class GpuVolume;
}

namespace gpu_voxel_constraints
{
    // Applies the constraint values to a displacement field in specified mask
    void constraint_displacement_field(GpuDisplacementField& df, const stk::GpuVolume& mask,
                                       const stk::GpuVolume& values);

    // Builds a volume pyramid for constraint mask and values
    void build_pyramids(const stk::GpuVolume& mask, const stk::GpuVolume& values,
        int num_levels, GpuVolumePyramid& mask_pyramid, GpuVolumePyramid& values_pyramid);
}
