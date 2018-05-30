#include <stk/image/volume.h>
#include <stk/io/io.h>

#include "deform_lib/arg_parser.h"
#include "deform_lib/registration/transform.h"

#include <iostream>

int run_transform(int argc, char* argv[])
{
    // Usage:
    // ./deform transform <src> <deformation> <out> [-i <nn/linear>]

    ArgParser args(argc, argv);
    args.add_positional("command", "registration, transform, regularize, jacobian");
    args.add_positional("source", "Path to the image you want to transform");
    args.add_positional("deformation", "Path to the deformation field used to transform");
    args.add_positional("output", "Path to the resulting file");

    args.add_option("interp", "-i, --interp", "Interpolation to use, either 'nn' or 'linear' (default)");

    if (!args.parse()) {
        return 1;
    }

    transform::Interp interp = transform::Interp_Linear;
    if (args.is_set("interp")) {
        if (args.option("interp") == "nn") {
            interp = transform::Interp_NN;
        }
        else if (args.option("interp") == "linear") {
            interp = transform::Interp_Linear;
        }
        else {
            std::cout << "Unrecognized interpolation option ('" << args.option("interp") << "')" 
                << std::endl << std::endl;
            args.print_help();
            return 1;
        }
    }

    stk::Volume src = stk::read_volume(args.positional("source").c_str());
    if (!src.valid())
        return 1;

    stk::Volume def = stk::read_volume(args.positional("deformation").c_str());
    if (!def.valid())
        return 1;
    ASSERT(def.voxel_type() == stk::Type_Float3);

    stk::Volume result = transform_volume(src, def, interp);
    stk::write_volume(args.positional("output").c_str(), result);
    
    return 0;
}