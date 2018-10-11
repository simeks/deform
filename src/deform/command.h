#pragma once

#include <stk/common/log.h>
#include <deform_lib/arg_parser.h>
#include <deform_lib/defer.h>
#include <deform_lib/version.h>

#include <deform_lib/profiler/profiler.h>

struct DeformCommand
{
    DeformCommand(int argc, char* argv[], const std::string& log_file)
        : _log_file{log_file}
        , _args(argc, argv)
        {}

    int execute(void) {
        if (!_parse_arguments()) {
            return -1;
        }

        if (_log_file != "") {
            stk::log_init();
            defer{stk::log_shutdown();};
            stk::log_add_file(_log_file.c_str(), stk::Info);
            LOG(Info) << "Version: " << deform::version_string();
        }

        return _execute();
    }

protected:
    virtual bool _parse_arguments(void) = 0;
    virtual int _execute(void) = 0;

    std::string _log_file;
    ArgParser _args;
};


struct RegistrationCommand : public DeformCommand
{
    RegistrationCommand(int argc, char* argv[], const std::string& log_file="deform_log.txt")
        : DeformCommand(argc, argv, log_file) {};
protected:
    virtual bool _parse_arguments(void);
    virtual int _execute(void);
};


struct TransformCommand : public DeformCommand
{
    TransformCommand(int argc, char* argv[]) : DeformCommand(argc, argv, "") {}
protected:
    virtual bool _parse_arguments(void);
    virtual int _execute(void);
};


struct RegularisationCommand : public DeformCommand
{
    RegularisationCommand(int argc, char* argv[]) : DeformCommand(argc, argv, "") {}
protected:
    virtual bool _parse_arguments(void);
    virtual int _execute(void);
};


struct JacobianCommand : public DeformCommand
{
    JacobianCommand(int argc, char* argv[]) : DeformCommand(argc, argv, "") {}
protected:
    virtual bool _parse_arguments(void) = 0;
    virtual int _execute(void) = 0;
};

