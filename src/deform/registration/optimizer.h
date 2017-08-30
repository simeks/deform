#pragma once

class Optimizer
{
public:
    virtual void execute(
        const Image* fixed, 
        const Image* moving, 
        int pair_count,
        ImageVec3d& def) = 0;
};

