#pragma once

class ArgParser
{
public:
    ArgParser();
    ~ArgParser();

    

    void add_option(const char* matchers, 
                    const char* name, 
                    const char* help)
    {

    }
 
    void parse(int argc, const char* argv[], int start_idx = 1);

};

namespace args
{
    // Print a line in the style of
    //  -m, --matcher name      help string
    inline void print_option_help(const char* matchers, const char* name, 
        const char* help)
    {
        std::string tmp = std::string(matchers) + " " + std::string(name);
        std::cout << std::string(4, ' ') << std::setw(30) << std::left << tmp 
                  << help << std::endl; 
    }

    // Print a line in the style of
    //  -f, --flag      help string
    inline void print_flag_help(const char* matchers, const char* help)
    {
        std::cout << std::string(4, ' ') << std::setw(30) << std::left 
                  << matchers << help << std::endl; 
    }
}
