#include "config_file.h"

#include <framework/debug/log.h>
#include <framework/types.h>

void exitWithError(const std::string &error)
{
    LOG(Error, error.c_str());
    exit(EXIT_FAILURE);
}

void ConfigFile::removeComment(std::string &line) const
{
    if (line.find("//") != line.npos)
        line.erase(line.find("//"));
}

bool ConfigFile::onlyWhitespace(const std::string &line) const
{
    return (line.find_first_not_of(' ') == line.npos);
}
bool ConfigFile::validLine(const std::string &line) const
{
    std::string temp = line;
    temp.erase(0, temp.find_first_not_of("\t "));
    if (temp[0] == '=')
        return false;

    for (size_t i = temp.find('=') + 1; i < temp.length(); i++)
        if (temp[i] != ' ')
            return true;

    return false;
}

void ConfigFile::extractKey(std::string &key, size_t const &sepPos, const std::string &line) const
{
    key = line.substr(0, sepPos);
    if (key.find('\t') != line.npos || key.find(' ') != line.npos)
        key.erase(key.find_first_of("\t "));
}
void ConfigFile::extractValue(std::string &value, size_t const &sepPos, const std::string &line) const
{
    value = line.substr(sepPos + 1);
    value.erase(0, value.find_first_not_of("\t "));
    value.erase(value.find_last_not_of("\t ") + 1);
}

void ConfigFile::extractContents(const std::string &line)
{
    std::string temp = line;
    temp.erase(0, temp.find_first_not_of("\t "));
    size_t sepPos = temp.find('=');

    std::string key, value;
    extractKey(key, sepPos, temp);
    extractValue(value, sepPos, temp);

    if (!keyExists(key))
        contents.insert(std::pair<std::string, std::string>(key, value));
    else
        exitWithError("CFG: Can only have unique key names!\n");
}

void ConfigFile::parseLine(const std::string &line, size_t const lineNo)
{
    if (line.find('=') == line.npos)
        exitWithError("CFG: Couldn't find separator on line: " + convert::T_to_string(lineNo) + "\n");

    if (!validLine(line))
        exitWithError("CFG: Bad format for line: " + convert::T_to_string(lineNo) + "\n");

    extractContents(line);
}

void ConfigFile::extractKeys()
{
    std::ifstream file;
    file.open(fName.c_str());
    if (!file)
        exitWithError("CFG: File " + fName + " couldn't be found!\n");

    std::string line;
    size_t lineNo = 0;
    while (std::getline(file, line))
    {
        lineNo++;
        std::string temp = line;

        if (temp.empty())
            continue;

        removeComment(temp);
        if (onlyWhitespace(temp))
            continue;

        parseLine(temp, lineNo);
    }

    file.close();
}


ConfigFile::ConfigFile(const std::string &fName)
{
    this->fName = fName;
    extractKeys();
}

bool ConfigFile::keyExists(const std::string &key) const
{
    return contents.find(key) != contents.end();
}

template <>
std::string convert::string_to_T(std::string const &val)
{
    return val;
}

template <>
int3 convert::string_to_T(std::string const& val)
{
    std::istringstream istr(val);
    int3 ret = {0};
    istr >> ret.x;
    istr >> ret.y;
    istr >> ret.z;

    return ret;
}