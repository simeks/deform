#include <stk/common/log.h>
#include <stk/image/volume.h>
#include <stk/io/io.h>

#include "deform_lib/arg_parser.h"
#include "deform_lib/jacobian.h"

#include "deform/command.h"

#include <iostream>

bool JacobianCommand::_parse_arguments(void)
{
    _args.add_positional("command", "registration, transform, regularize, jacobian");
    _args.add_positional("deformation", "Path to the deformation field");
    _args.add_positional("output", "Path to the resulting file");
    return _args.parse();
}

int JacobianCommand::_execute(void)
{
    LOG(Info) << "Computing jacobian.";
    LOG(Info) << "Input: '" << _args.positional("deformation") << "'";

    stk::Volume def = stk::read_volume(_args.positional("deformation").c_str());
    if (!def.valid()) {
        return EXIT_FAILURE;
    }

    stk::Volume jac = calculate_jacobian(def);

    LOG(Info) << "Writing to '" << _args.positional("output") << "'";
    stk::write_volume(_args.positional("output").c_str(), jac);

    return EXIT_SUCCESS;
}

