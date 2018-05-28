#include <stk/image/volume.h>
#include <stk/io/io.h>

namespace
{
    void print_help(const char* exec)
    {
        std::cout << "Usage: " << exec << " transform [source] [deformation] [output] {OPTIONS}" 
            << std::endl << std::endl;
        
        std::cout << "OPTIONS:" << std::endl << std::endl;

    }
}

int run_transform(int argc, char* argv[])
{
    // Usage:
    // ./deform transform <src> <deformation> <out> [-i <nn/linear>]

    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " transform <src> <deformation> <out> [-i <nn/linear>]" << std::endl;
        return 1;
    }

    stk::Volume src = stk::read_volume(argv[2]);
    if (!src.valid())
        return 1;

    stk::Volume def = stk::read_volume(argv[3]);
    if (!def.valid())
        return 1;

    transform::Interp interp = transform::Interp_Linear;

    // TODO: Quick fix, include when refactoring command-line args
    if (argc == 7 && strcmp(argv[5], "-i") == 0 && strcmp(argv[6], "nn") == 0) {
        interp = transform::Interp_NN;
    }

    // TODO: Verify that def is float3

    stk::Volume result = transform_volume(src, def, interp);
    stk::write_volume(argv[4], result);
    
    return 0;
}