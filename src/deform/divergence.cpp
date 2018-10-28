#include <stk/common/log.h>
#include <stk/filters/vector_calculus.h>
#include <stk/io/io.h>

#include "deform_lib/arg_parser.h"

#include "deform/command.h"

bool DivergenceCommand::_parse_arguments(void)
{
    _args.add_positional("command", "registration, transform, regularize, jacobian");
    _args.add_positional("displacement", "Path to the displacement field");
    _args.add_positional("output", "Path to the resulting file");
    return _args.parse();
}

int DivergenceCommand::_execute(void)
{
    LOG(Info) << "Computing divergence.";
    LOG(Info) << "Input: '" << _args.positional("displacement") << "'";

    stk::Volume def = stk::read_volume(_args.positional("displacement").c_str());
    if (!def.valid()) {
        return EXIT_FAILURE;
    }

    stk::Volume div = stk::divergence<float3>(def);

    LOG(Info) << "Writing to '" << _args.positional("output") << "'";
    stk::write_volume(_args.positional("output").c_str(), div);

    return EXIT_SUCCESS;
}

