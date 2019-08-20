#include "arg_parser.h"

#include <stk/common/assert.h>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>

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
    add_flag("help", "-h, --help", "Displays this help text");
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
                           const char* help,
                           bool required)
{
    ASSERT(!check_collision(name));
    _options.push_back({
        name,
        matchers,
        help,
        (int)_groups.size()-1,
        false,
        required
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
        true,
        false // Flags are never required
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
    ASSERT(_option_values.empty()); // Should only be run once

    _positionals[0].read = _argv[0];
    size_t positional_idx = 1;

    int i = 1; // Skip argv[0]
    while (i < _argc) {
        if (_argv[i][0] == '-') {
            if (!parse_arg(i))
                break;
        }
        else {
            // Positional
            if (positional_idx >= _positionals.size()) {
                _error << "Unexpected postional";
                break;
            }
            _positionals[positional_idx].read = _argv[i];
            ++positional_idx;
        }
        ++i;
    }

    if (error() != "") {
        std::cout << error() << std::endl << std::endl;
        print_help();
        return false;
    }

    // Allow --help to print help section before any input validation
    if (is_set("help")) {
        print_help();
        return false;
    }

    // Validate input

    // Check if we have all the positionals
    if (positional_idx < _positionals.size()) {
        _error << "Missing arguments: '"
               << _positionals[positional_idx].name << "'";

        for (auto p = positional_idx+1; p < _positionals.size(); ++p) {
            _error << ", '" << _positionals[p].name << "'";
        }
    }

    // Check if we have all required options
    for (auto& opt : _options) {
        if (opt.required) {
            // Check if options is an array option
            std::string name = opt.name;
            // --arg{i}
            size_t idx;
            if ((idx = name.find("{i}")) != std::string::npos) {
                name = name.substr(0, name.size()-3);
                // If an array-option is required, we require at least one value
                auto it = std::find_if(_option_values.begin(), _option_values.end(),
                    [&](const std::pair<std::string, std::string>& v){
                        return v.first.compare(0, name.size(), name) == 0;
                    });
                if (it == _option_values.end()) {
                    _error << "Missing required argument '" << name << "{i}'";
                    break;
                }
            }
            else {
                if (_option_values.find(name) == _option_values.end()) {
                    _error << "Missing required argument '" << name << "'";
                    break;
                }
            }
        }
    }

    if (error() != "") {
        std::cout << error() << std::endl << std::endl;
        print_help();
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
    for (size_t p = 1; p < _positionals.size(); ++p) {
        std::cout << " [" << _positionals[p].name << "]";
    }

    std::cout << std::endl << std::endl;

    std::string indent(4, ' ');
    for (size_t p = 1; p < _positionals.size(); ++p) {
        std::cout << indent << std::setw(40) << std::left << _positionals[p].name
            << _positionals[p].help << std::endl;
    }

    std::cout << std::endl;
    std::cout << "OPTIONS:" << std::endl;

    for (int g = 0; g < (int) _groups.size(); ++g) {
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
    return 0;
}
template<>
std::string ArgParser::get<std::string>(const std::string& name, std::string def)
{
    if (is_set(name)) {
        return option(name);
    }
    return def;
}
template<>
int ArgParser::get<int>(const std::string& name, int def)
{
    if (is_set(name)) {
        return std::stoi(option(name));
    }
    return def;
}
template<>
float ArgParser::get<float>(const std::string& name, float def)
{
    if (is_set(name)) {
        return std::stof(option(name));
    }
    return def;
}
template<>
double ArgParser::get<double>(const std::string& name, double def)
{
    if (is_set(name)) {
        return std::stod(option(name));
    }
    return def;
}

std::string ArgParser::positional(int i)
{
    ASSERT(i < (int) _positionals.size());
    ASSERT(_positionals[i].read);
    return _positionals[i].read;
}

int ArgParser::count_instances(const std::string& name) const
{
    int count = 0;

    size_t idx;
    if ((idx = name.find("{i}")) != std::string::npos) {
        std::string name2 = name.substr(0, name.size()-3);
        
        for (auto& ov : _option_values) {
            if (ov.first.compare(0, name2.size(), name2) == 0) {
                ++count;
            }
        }
    }
    else {
        if (_option_values.find(name) != _option_values.end()) {
            ++count;
        }
    }

    return count;
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

std::vector<std::string> ArgParser::split_matchers(const char* in) const
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
                    try {
                        id = std::stoi(id_str, &end_index);
                    } catch (std::invalid_argument&) {
                        // Another option may share a prefix
                        return false;
                    }
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
