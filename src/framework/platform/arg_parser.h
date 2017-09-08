#pragma once

#include <map>
#include <string>
#include <vector>

/// Parses command line arguments on the format
/// executable.exe <token0> <token1> -<key0>=<value0> --<key1>=<value1> <token2>
class ArgParser
{
public:
    /// argc and argv as passed to the program in main()
    ArgParser(int argc, char** argv);
    ~ArgParser();

    /// Checks if the given key is set (e.g. --<key>=<value>)
    bool is_set(const std::string& key) const;

    /// Returns the value for the given key, fails if key doesn't exist
    const std::string& value(const std::string& key) const;

    /// Returns a map of all key-values
    const std::map<std::string, std::string>& values() const;
    
    /// Returns token with index i
    const std::string& token(int i) const;

    /// Returns the number of tokens
    int num_tokens() const;

    /// Returns the name of the executable
    const std::string& executable() const;

private:
    std::string _executable;
    std::map<std::string, std::string> _values;
    std::vector<std::string> _tokens;

};
