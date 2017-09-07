#pragma once

class Volume;
class Optimizer
{
public:
    virtual void execute(
        Volume* fixed, 
        Volume* moving, 
        int pair_count,
        Volume& def) = 0;
};

