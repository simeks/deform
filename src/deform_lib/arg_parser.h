#pragma once

#include <map>
#include <sstream>
#include <vector>

// Utility for parsing argc, argv-style command-line arguments
//
// Example usage:
//  ArgParser parser(argc, argv);
//  parser.add_group("Group1");
//  parser.add_positional("command", "Command to perform");
//  parser.add_option("image{i}", "-i{i}, --image{i}", "Image file for index i");
//  parser.add_option("output", "-o, --output", "Output file");
//  parser.add_group("Flags");
//  parser.add_flag("flag", "-f, --flag", "Flag");
//
//  if (!parser.parse()) {
//      std::cout << parser.error() << std::endl;
//      parser.print_help();
//      return 1;
//  }
//  if (parser.positional("command") == "add") {
//      bool flag = parser.is_set("flag");
//      if (parser.is_set("file0")) {
//          read_file(parser.option("file0"));
//      }
//      ...
//  }
//
// User begin by adding groups, options, flags, and positionals. Order of these
//  add_* calls are important as that defines the order it will be presented in
//  the help section. When adding a group, all sequent options will be added to
//  the group until a new group is added.
//
// There are two types of options, named and flags. Named options are a typical
//  key=value pair, which would be specified as "--key value" in the command-
//  line. Flags are either true (set), or false (not set).
//
// Options are added through add_option and add_flag, with the same syntax.
//  A name, which is the identifier for the option and will be used later when
//  accessing the data through is_set() and option(), a set of matchers, which
//  is the matchers to look for when parsing the command-line, and a help text.
//  It is possible to allow multiple matchers for a single option by seperating
//  each with ',', E.g. "-f, --flag", will allow both -f and --flag in the 
//  command-line. Matchers are required to start with either '-' or '--'.
//
// It is also possible to add array options. This way an user can add an
//  arbitrary number of options within a specific key. An example:
//  "program.exe -i0 image0 -i1 image1 -i2 image2"
//  Array options are specified as a regular option but with a {i} added to the
//  identifier and the matchers, the parser will automatically evaluate the {i}
//  and add the read option value as "i0", "i1", "i2", etc. To read an option
//  simply call option("i0").
// Using array options, with the matcher "-i{i}", a call with "-i something" 
//  will result in "something" being stored in the key "i0", as the default 
//  index is 0.
//
// Positional arguments are arguments not specified through matchers but rather
//  their position within the command-line. The order of add_positional calls
//  determine the order the parser expect them to appear in the command-line.
//  An example: "program.exe positional1 positional2 ..."
//
// Parser automatically handles -h and --help flags, showing the help section.
class ArgParser
{
public:
    ArgParser(int argc, char* argv[]);
    ~ArgParser();

    // Adds a option group, purely for aesthetics, groups do not affect parsing.
    //  An empty name will only show up as a line-break in the help section.
    void add_group(const char* name = "");

    // Adds an option, name will be used as the identifier when later on using
    //  is_set() and option(). 'matchers' will specifiy a set of matchers to
    //  look for when parsing, example: "-f, --flag". 'help' will be displayed 
    //  when displaying the help section.
    void add_option(const char* name, const char* matchers, 
                    const char* help, bool required=false);
    // See add_option
    void add_flag(const char* name, const char* matchers, 
                  const char* help);
    // Adds a positional argument, name will later be used when calling 
    //  positional(), help is displayed in help section.
    void add_positional(const char* name, const char* help);

    // Parses the command-line arguments given in the constructor.
    //  Returns true if parsing succeeded, false if not.
    // Use error() to figure out what went wrong if parsing failed. 
    bool parse();

    // Prints a help section of all added options.
    void print_help();

    // Returns true if the given option is set, could be either a flag or a 
    //  key-value
    bool is_set(const std::string& name);

    // Check if the parser holds a positional of the given name
    bool has_positional(const std::string& name);

    // Returns the value of the given option, assuming it is set.
    std::string option(const std::string& name);

    // Returns the value of the given positional, assuming it is set.
    std::string positional(const std::string& name);

    // Returns the positional at the given index. i=0 will always point to the
    //  name of the executable.
    std::string positional(int i);

    // Returns the last error.
    std::string error();

    template<typename T>
    T get(const std::string& name, T def);

private:
    struct Option
    {
        const char* name;
        const char* matchers;
        const char* help;
        int group;
        bool flag;
        bool required;
    };

    struct Positional
    {
        const char* name;
        const char* help;
        const char* read; // 0 if no data was read
    };

    struct Group
    {
        const char* name;
    };

    // Check if we already have an option or positional with the specified name
    // Returns true if an object with the name already exists
    bool check_collision(const char* name);

    // Splits a string of matchers ("-f, --flag, --flag2") to a vector of strings
    std::vector<std::string> split_matchers(const char* in);

    // Returns true if the specified option matches the given option, false if not
    // If a match is found, the key in which to store the command-line argument is 
    //  return through argument 'key'.
    bool match_option(const Option& opt, const std::string& arg, std::string& key);

    // Parse arg at argv[i]
    bool parse_arg(int& i);

    void set_flag(const std::string& name);
    void set_option(const std::string& name, const std::string& value);

    int _argc;
    char** _argv;

    std::vector<Option> _options;
    std::vector<Positional> _positionals;
    std::vector<Group> _groups;

    std::map<std::string, std::string> _option_values;

    std::stringstream _error;
};
