#include <deform_lib/version.h>

#include <sstream>

std::string deform::version_string()
{
    std::stringstream ss;
    ss << GIT_VERSION_TAG << "-" << GIT_SHA1_SHORT << (GIT_DIRTY ? "+" : "");

    #ifdef _DEBUG
        ss << " [DEBUG]";
    #endif

    return ss.str();
}

std::string deform::short_version_string()
{
    std::stringstream ss;
    ss << "v" << GIT_VERSION_TAG;

    #ifdef _DEBUG
        ss << " [DEBUG]";
    #endif

    return ss.str();
}


