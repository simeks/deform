#include <stk/image/volume.h>
#include <stk/io/io.h>

#include "deform_lib/arg_parser.h"
#include "deform_lib/jacobian.h"

#include <iostream>

int run_jacobian(int argc, char* argv[])
{
    // Usage:
    // ./deform jacobian <source> <deformation> <out>

    ArgParser args(argc, argv);
    args.add_positional("command", "registration, transform, regularize, jacobian");
    args.add_positional("source", "Path to the source (moving) image, required for the spacing/origin");
    args.add_positional("deformation", "Path to the deformation field");
    args.add_positional("output", "Path to the resulting file");

    if (!args.parse()) {
        return 1;
    }

    stk::Volume src = stk::read_volume(args.positional("source").c_str());
    if (!src.valid())
        return 1;

    stk::Volume def = stk::read_volume(args.positional("deformation").c_str());
    if (!def.valid())
        return 1;

    stk::Volume jac = calculate_jacobian(src, def);
    stk::write_volume(args.positional("output").c_str(), jac);
    
    return 0;
}
