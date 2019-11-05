#include <stk/common/log.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>

#include "deform_lib/arg_parser.h"
#include "deform_lib/registration/transform.h"

#include "deform/command.h"

bool TransformCommand::_parse_arguments(void)
{
    _args.add_positional("command", "registration, transform, regularize, jacobian");
    _args.add_positional("source", "Path to the image you want to transform");
    _args.add_positional("transform", "Path to the transform file, could be either affine or a displacement field");
    _args.add_positional("output", "Path to the resulting file");

    _args.add_option("interp", "-i, --interp", "Interpolation to use, either 'nn' or 'linear' (default)");

    return _args.parse();
}

int TransformCommand::_execute(void)
{
    LOG(Info) << "Transforming volume";

    transform::Interp interp = transform::Interp_Linear;
    if (_args.is_set("interp")) {
        if (_args.option("interp") == "nn") {
            interp = transform::Interp_NN;
        }
        else if (_args.option("interp") == "linear") {
            interp = transform::Interp_Linear;
        }
        else {
            std::cout << "Unrecognized interpolation option ('" << _args.option("interp") << "')"
                << std::endl << std::endl;
            _args.print_help();
            return EXIT_FAILURE;
        }
    }

    LOG(Info) << "Interpolation method: " << ((interp == transform::Interp_Linear) ? "linear" : "nn");
    LOG(Info) << "Input: '" << _args.positional("source") << "'";
    LOG(Info) << "Displacement: '" << _args.positional("displacement") << "'";

    stk::Volume src = stk::read_volume(_args.positional("source").c_str());
    if (!src.valid())
        return EXIT_FAILURE;

    stk::Volume def = stk::read_volume(_args.positional("displacement").c_str());
    if (!def.valid())
        return EXIT_FAILURE;
    ASSERT(def.voxel_type() == stk::Type_Float3);

    stk::Volume result = transform_volume(src, def, interp);
    if (!result.valid())
        return EXIT_FAILURE;

    LOG(Info) << "Writing to '" << _args.positional("output") << "'";
    stk::write_volume(_args.positional("output").c_str(), result);

    return EXIT_SUCCESS;
}
