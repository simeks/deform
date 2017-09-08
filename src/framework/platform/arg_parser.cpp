#include "arg_parser.h"

#include "debug/assert.h"

ArgParser::ArgParser(int argc, char** argv)
{
    _executable = argv[0];

    std::vector<std::string> tokens;
    for (int i = 1; i < argc; ++i)
        tokens.push_back(argv[i]);

    while (!tokens.empty())
    {
        const std::string& token = tokens.back();
        
        if (token[0] == '-')
        {
            int b = 1;
            if (token[1] == '-')
            {
                b = 2;
            }

            std::string line = token.substr(b);
            size_t p = line.find('=');
            if (p != std::string::npos)
            {
                std::string key = line.substr(0, p);
                std::string value = line.substr(p + 1);
                _values[key] = value;
            }
            else
            {
                _values[line] = "";
            }
        }
        else
        {
            _tokens.push_back(token);
        }
        tokens.pop_back();
    }
}
ArgParser::~ArgParser()
{
}

bool ArgParser::is_set(const std::string& key) const
{
    return _values.find(key) != _values.end();
}
const std::string& ArgParser::value(const std::string& key) const
{
    assert(is_set(key));
    return _values.at(key);
}
const std::map<std::string, std::string>& ArgParser::values() const
{
    return _values;
}

const std::string& ArgParser::token(int i) const
{
    return _tokens[i];
}
int ArgParser::num_tokens() const
{
    return (int)_tokens.size();
}
const std::string& ArgParser::executable() const
{
    return _executable;
}
