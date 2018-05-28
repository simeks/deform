#include <stk/image/volume.h>
#include <stk/io/io.h>

#include "deform_lib/jacobian.h"

int run_jacobian(int argc, char* argv[])
{
    // Usage:
    // ./deform jacobian <source> <deformation> <out>

    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " jacobian [SOURCE] [DEFORMATION] [OUTPUT]" << std::endl;
        return 1;
    }

    stk::Volume src = stk::read_volume(argv[2]);
    if (!src.valid())
        return 1;

    stk::Volume def = stk::read_volume(argv[3]);
    if (!def.valid())
        return 1;

    stk::Volume jac = calculate_jacobian(src, def);
    stk::write_volume(argv[4], jac);
    
    return 0;
}
