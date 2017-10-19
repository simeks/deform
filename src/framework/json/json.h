#pragma once

#include <stdint.h>
#include <string>

class JsonObject;

class JsonReader
{
public:
    JsonReader();
    ~JsonReader();

    /// Parses a JSON document into JsonObject
    /// @param doc JSON docoument
    /// @param root This is going to be the root node
    /// @return True if the parsing was successful, else false
    bool read(const char* doc, int64_t length, JsonObject& root);

    /// Reads a JSON document from a file
    /// @return True if the parsing was successful, else false
    bool read_file(const std::string& file_name, JsonObject& root);

    /// Returns an error message if the last call to Parse failed.
    const std::string& error_message();


private:
    const char* _begin;
    const char* _cur;
    const char* _end;

    std::string _error;

    void error(const char* msg);
    void current_position(int& line, int& column);

    bool parse_value(JsonObject& value);
    bool parse_number(JsonObject& value);
    bool parse_double(JsonObject& value);
    bool parse_array(JsonObject& value);
    bool parse_object(JsonObject& value);

    bool parse_string(std::string& str);
    void skip_spaces();

};

class JsonWriter
{
public:
    JsonWriter();
    ~JsonWriter();

    /// @brief Generates JSON from the specified JsonObject
    /// @param root Root config node
    /// @param out The generated JSON will be stored in this variable
    /// @param format Should we use any formatting. Formatting makes it more 
    ///               human readable, otherwise everything is just printed on one line.
    void write(const JsonObject& root, std::stringstream& out, bool format);

    /// @brief Writes a JSON file from the specified JsonObject
    /// @param root Root config node
    /// @param file_name File name of the destination
    /// @param format Should we use any formatting. Formatting makes it more 
    ///               human readable, otherwise everything is just printed on one line.
    /// @return True if the parsing was successful, else false
    bool write_file(const JsonObject& root, const std::string& file_name, bool format);

private:
    bool _format;
    int _ilevel; // Indent level

    void write_value(const JsonObject& node, std::ostream& out);

    /// @param quotes Specifies whetever the string is supposed to be surrounded by quotes
    void write_string(const std::string& str, std::ostream& out, bool quotes = true);

    void write_tabs(int ilevel, std::ostream& out);

};


