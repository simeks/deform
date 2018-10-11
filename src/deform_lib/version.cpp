#include <deform_lib/version.h>

#include <sstream>

std::string deform::version_string(void)
{
    std::stringstream ss;
    ss << GIT_SHA1_SHORT << "@" << GIT_BRANCH << (GIT_DIRTY ? "+" : "") << " (" << GIT_DATE << ")";

    #ifdef _DEBUG
        ss << " [DEBUG]";
    #endif

    return ss.str();
}

