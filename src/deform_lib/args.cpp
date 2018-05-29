#include "args.h"

#include <stk/common/assert.h>

#include <algorithm>
#include <iomanip>
#include <iostream>

ArgParser::ArgParser(int argc, char* argv[]) :
    _argc(argc),
    _argv(argv)
{
    _groups.push_back({""});
    _positionals.push_back({ // For argv[0]
        0,
        0,
        0
    });
}
ArgParser::~ArgParser() {}

void ArgParser::add_group(const char* name)
{
    _groups.push_back({
        name
    });
}
void ArgParser::add_option(const char* name, 
                           const char* matchers, 
                           const char* help)
{
    ASSERT(!check_collision(name));
    _options.push_back({
        name,
        matchers,
        help,
        (int)_groups.size()-1,
        false
    });
}
void ArgParser::add_flag(const char* name, 
                         const char* matchers, 
                         const char* help)
{
    ASSERT(!check_collision(name));
    _options.push_back({
        name,
        matchers,
        help,
        (int)_groups.size()-1,
        true
    });
}
void ArgParser::add_positional(const char* name, const char* help)
{
    ASSERT(!check_collision(name));
    _positionals.push_back({
        name,
        help,
        0
    });
}
bool ArgParser::parse()
{
    _positionals[0].read = _argv[0];
    int positional_idx = 1;

    int i = 1; // Skip argv[0]
    while (i < _argc) {
        if (_argv[i][0] == '-') {
            if (!parse_arg(i))
                return false;
        }
        else {
            // Positional
            if (positional_idx >= _positionals.size()) {
                _error << "Unexpected postional";
                return false;
            }
            _positionals[positional_idx].read = _argv[i];
            ++positional_idx;
        }
        ++i;
    }
    if (positional_idx < _positionals.size()) {
        _error << "Missing positional arguments: '" 
               << _positionals[positional_idx].name << "'";
        
        for (int p = positional_idx+1; p < _positionals.size(); ++p) {
            _error << ", " << _positionals[p].name;
        }
        return false;
    }
    return true;
}
void ArgParser::print_help()
{
    std::cout << "Usage: " << _argv[0];
    if (_options.size()) {
        std::cout << " {OPTIONS}";
    }

    // Skip argv[0]
    for (int p = 1; p < _positionals.size(); ++p) {
        std::cout << " [" << _positionals[p].name << "]";
    }

    std::cout << std::endl << std::endl;

    std::string indent(4, ' ');
    for (int p = 1; p < _positionals.size(); ++p) {
        std::cout << indent << std::setw(40) << std::left << _positionals[p].name 
            << _positionals[p].help << std::endl; 
    }

    std::cout << std::endl;
    std::cout << "OPTIONS:" << std::endl;

    for (int g = 0; g < _groups.size(); ++g) {
        if (strcmp(_groups[g].name, "") != 0) {
            std::cout << indent << _groups[g].name << ":" << std::endl;
        }

        for (auto& o : _options) {
            if (o.group == g) {
                if (o.flag) {
                    std::cout << indent << std::setw(40) << std::left << o.matchers 
                            << o.help << std::endl;
                }
                else {
                    std::string tmp = std::string(o.matchers) + " " + std::string(o.name);
                    std::cout << std::string(4, ' ') << std::setw(40) << std::left << tmp 
                            << o.help << std::endl;
                }
            }
        }
        std::cout << std::endl;
    }
}

bool ArgParser::is_set(const std::string& name)
{
    return _option_values.find(name) != _option_values.end();
}

std::string ArgParser::option(const std::string& name)
{
    ASSERT(is_set(name));
    return _option_values[name];
}

std::string ArgParser::positional(const std::string& name)
{
    for (auto& p : _positionals) {
        if (p.name && name == p.name) {
            ASSERT(p.read);
            return p.read;
        }
    }
    ASSERT(false);
    return "";
}

std::string ArgParser::positional(int i)
{
    ASSERT(i < _positionals.size());
    ASSERT(_positionals[i].read);
    return _positionals[i].read;
}

std::string ArgParser::error()
{
    return _error.str();
}

bool ArgParser::check_collision(const char* name)
{
    for (auto& opt : _options) {
        if (opt.name && strcmp(opt.name, name) == 0) return true;
    }
    for (auto& p : _positionals) {
        if (p.name && strcmp(p.name, name) == 0) return true;
    }
    return false;
}

std::vector<std::string> ArgParser::split_matchers(const char* in)
{
    std::vector<std::string> results;
    std::string tmp = in;

    // Remove whitespace
    tmp.erase(std::remove_if(tmp.begin(), tmp.end(), ::isspace), tmp.end());

    size_t last = 0;
    size_t next = 0;
    while ((next = tmp.find(',', last)) != std::string::npos) {
        std::string token = tmp.substr(last, next-last);
        last = next + 1;

        results.push_back(token);
    }
    results.push_back(tmp.substr(last));
    return results;
}

bool ArgParser::match_option(const Option& opt, const std::string& arg, std::string& key)
{
    std::vector<std::string> matchers = split_matchers(opt.matchers);

    for (auto& m : matchers) {
        std::string token = m;
        // --arg{i}
        size_t idx;
        if ((idx = token.find("{i}")) != std::string::npos) {
            token = token.substr(0, idx); // --arg

            // Handle the special case for arrays
            // Assumes the arg ends with {i}

            if (arg.compare(0, token.size(), token) == 0) { // --arg vs --arg123
                // The arg is only a match if the rest is numbers, no number
                //  will be the same as --arg0

                std::string id_str = arg.substr(token.size()); // --arg123 => 123
                
                size_t end_index = id_str.size();
                int id = 0;
                if (id_str.size()) {
                    id = std::stoi(id_str, &end_index);
                }

                // Make sure the integer ends the string, so we don't get weird
                //  cases with parsing --asd123fgh as --asd[123]
                if (end_index == id_str.size()) {
                    std::string tmp = opt.name;
                    tmp = tmp.substr(0, tmp.size()-3); // Remove {i}

                    std::stringstream ss;
                    ss << tmp << id;
                    key = ss.str();
                    return true;
                }
            }
        }

        // --arg
        if (arg == token) {
            key = opt.name;
            return true;
        }
    }
    return false;
}

bool ArgParser::parse_arg(int& i)
{
    for (auto& opt : _options) {
        std::string key;
        if (match_option(opt, _argv[i], key)) {
            if (_option_values.find(key) != _option_values.end()) {
                _error << "Already set: " << _argv[i];
                return false;
            }

            if (opt.flag) {
                set_flag(key);
            }
            else {
                if (++i >= _argc) {
                    _error << "Missing arguments for '" << _argv[i-1] << "'";
                    return false;
                }
                set_option(key, _argv[i]);
            }
            return true;
        }
    }
    _error << "Unrecognized option: " << _argv[i];
    return false;
}

void ArgParser::set_flag(const std::string& name)
{
    _option_values[name] = "true";
}
void ArgParser::set_option(const std::string& name, const std::string& value)
{
    _option_values[name] = value;
}
