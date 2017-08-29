#pragma once

namespace log
{
    enum Level
    {
        Info,
        Warning,
        Error, // Typically asserts and such
        Fatal, // Error that kills the application

        NotImplemented
    };

    // TODO: Extend logging
}

#define LOG(level, fmt, ...) printf(fmt, __VA_ARGS__)
