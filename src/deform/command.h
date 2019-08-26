#pragma once

#include <stk/common/log.h>
#include <deform_lib/arg_parser.h>
#include <deform_lib/defer.h>
#include <deform_lib/version.h>

#include <deform_lib/profiler/profiler.h>

#include <iostream>

struct DeformCommand
{
    DeformCommand(int argc, char* argv[], bool log_to_file = false)
        : _args(argc, argv)
        , _log_to_file{log_to_file}
    {}

    virtual ~DeformCommand() {
        stk::log_shutdown();
    }

    int execute(void) {
        if (!_parse_arguments()) {
            return -1;
        }

        stk::log_init();

        // Read log level
        const char * const log_level_p = std::getenv("DF_LOG_LEVEL");
        stk::LogLevel log_level = stk::LogLevel::Info;
        if (log_level_p) {
            try {
                log_level = stk::log_level_from_str(log_level_p);
            }
            catch (const std::runtime_error&) {
                LOG(Error) << "Invalid value for the environment variable " <<
                              "DF_LOG_LEVEL=' " << std::string(log_level_p);
                return EXIT_FAILURE;
            }
        }

        stk::log_add_stream(&std::cerr, log_level);

        if (_log_to_file) {
            const char * const log_file_p = std::getenv("DF_LOG_FILE");
            const std::string log_file {log_file_p ? log_file_p : "deform_log.txt"};

            stk::log_add_file(log_file.c_str(), log_level);
            LOG(Info) << "Version: " << deform::version_string();
        }

        return _execute();
    }

protected:
    virtual bool _parse_arguments(void) = 0;
    virtual int _execute(void) = 0;

    ArgParser _args;
    const bool _log_to_file;
};


struct RegistrationCommand : public DeformCommand
{
    using DeformCommand::DeformCommand;
protected:
    virtual bool _parse_arguments(void);
    virtual int _execute(void);
};


struct TransformCommand : public DeformCommand
{
    using DeformCommand::DeformCommand;
protected:
    virtual bool _parse_arguments(void);
    virtual int _execute(void);
};


struct RegularisationCommand : public DeformCommand
{
    using DeformCommand::DeformCommand;
protected:
    virtual bool _parse_arguments(void);
    virtual int _execute(void);
};


struct JacobianCommand : public DeformCommand
{
    using DeformCommand::DeformCommand;
protected:
    virtual bool _parse_arguments(void);
    virtual int _execute(void);
};


struct DivergenceCommand : public DeformCommand
{
    using DeformCommand::DeformCommand;
protected:
    virtual bool _parse_arguments(void);
    virtual int _execute(void);
};

