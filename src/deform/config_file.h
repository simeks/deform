#pragma once

#include <framework/debug/log.h>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace convert
{
    template <typename T>
    std::string T_to_string(T const &val)
    {
        std::ostringstream ostr;
        ostr << val;

        return ostr.str();
    }

    template <typename T>
    T string_to_T(std::string const &val)
    {
        std::istringstream istr(val);
        T returnVal;
        if (!(istr >> returnVal))
        {
            LOG(Error, "CFG: Not a valid %s received!\n", typeid(T).name());
            exit(1);
        }

        return returnVal;
    }

} // namespace convert

class ConfigFile
{
private:
    std::map<std::string, std::string> contents;
    std::string fName;

    void removeComment(std::string &line) const;
    
    bool onlyWhitespace(const std::string &line) const;
    
    bool validLine(const std::string &line) const;
    
    void extractKey(std::string &key, size_t const &sepPos, const std::string &line) const;
    void extractValue(std::string &value, size_t const &sepPos, const std::string &line) const;
    void extractContents(const std::string &line);
    
    void parseLine(const std::string &line, size_t const lineNo);
    void extractKeys();
public:
    ConfigFile(const std::string &fName);
    bool keyExists(const std::string &key) const;

    template <typename ValueType>
    ValueType getValueOfKey(const std::string &key, ValueType const &defaultValue = ValueType()) const
    {
        if (!keyExists(key))
            return defaultValue;

        return convert::string_to_T<ValueType>(contents.find(key)->second);
    }
};

