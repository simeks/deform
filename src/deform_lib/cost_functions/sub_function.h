#pragma once

#include <stk/math/float3.h>
#include <stk/math/int3.h>
#include <stk/image/volume.h>

struct SubFunction
{
    /*!
     * \brief Cost term for a single voxel.
     * \param p Indices of the voxel in the reference image.
     * \param def Tentative displacement for the voxel.
     * \return The cost associated to displacing `p` by `def`.
     */
    virtual float cost(const int3& p, const float3& def) = 0;

    /*!
     * \brief A callback that is executed before each iteration of the solver.
     * \param iteration Number of the iteration just completed.
     * \param def Deformation field at the end of the iteration.
     */
    virtual void pre_iteration_hook(const int iteration, const stk::VolumeFloat3& def)
    {
        (void) iteration;
        (void) def;
    }
};

