#include <iomanip>
#include <iostream>
#include <string>

namespace {
void print_command_help(const char* exec)
{
    std::cout << "Usage: " << exec << " COMMAND ..." << std::endl << std::endl;
    std::cout << "COMMANDS:" << std::endl << std::endl;

    const char* commands[] = {
        "cost", "Computes cost for a given input and parameter set",
        nullptr, nullptr
    };

    int i = 0;
    while(commands[i] != nullptr) {
        std::cout << std::string(4, ' ') << std::setw(30) << std::left 
                  << commands[i] << commands[i+1] << std::endl;
        i += 2;
    }
}
}

int run_cost(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    if (argc >= 2 && strcmp(argv[1], "cost") == 0)
        return run_cost(argc, argv);
    
    print_command_help(argv[0]);

    return 1;
}
