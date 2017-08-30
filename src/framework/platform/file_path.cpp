#include "file_path.h"

#ifdef DF_PLATFORM_WINDOWS
#define PATH_SEPARATOR '\\'
#else
#define PATH_SEPARATOR '/'
#endif

FilePath::FilePath()
{
}
FilePath::FilePath(const char* path) : _path(path)
{
}
FilePath::FilePath(const std::string& path) : _path(path)
{
}
FilePath::~FilePath()
{
}

void FilePath::set(const char* path)
{
    _path = path;
}

void FilePath::set(const std::string& path)
{
    _path = path;
}
void FilePath::clear()
{
    _path.clear();
}
void FilePath::join(const char* path)
{
    *this += PATH_SEPARATOR;
    *this += path;
}
void FilePath::join(const std::string& path)
{
    *this += PATH_SEPARATOR;
    *this += path;
}
void FilePath::set_separator(char c)
{
    char* path = &_path[0];
    while (*path)
    {
        if (*path == '\\' || *path == '/')
        {
            *path = c;
        }
        path++;
    }
}
void FilePath::trim_extension()
{
    size_t pos = _path.rfind('.');
    if (pos != std::string::npos)
    {
        _path.erase(pos);
    }
}

std::string FilePath::directory() const
{
    size_t pos = _path.rfind('\\');
    if (pos == std::string::npos) // Separator may be either '/' or '\\'
        pos = _path.rfind('/');
    if (pos != std::string::npos)
    {
        return _path.substr(0, pos + 1);
    }
    return "";
}
std::string FilePath::filename() const
{
    size_t pos = _path.rfind('\\');
    if (pos == std::string::npos) // Separator may be either '/' or '\\'
        pos = _path.rfind('/');
    if (pos != std::string::npos)
    {
        return _path.substr(pos + 1);
    }
    return _path;
}
std::string FilePath::extension() const
{
    size_t pos = _path.rfind('.');
    if (pos != std::string::npos)
    {
        return _path.substr(pos + 1);
    }
    return "";
}
std::string& FilePath::get()
{
    return _path;
}
const std::string& FilePath::get() const
{
    return _path;
}
const char* FilePath::c_str() const
{
    return _path.c_str();
}

FilePath& FilePath::operator+=(const std::string& other)
{
    _path += other;
    return *this;
}
FilePath& FilePath::operator+=(const char* other)
{
    _path += other;
    return *this;
}
FilePath& FilePath::operator+=(const char other)
{
    _path += other;
    return *this;
}

