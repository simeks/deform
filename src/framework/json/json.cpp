#include "json.h"
#include "json_object.h"

#include <algorithm>
#include <fstream>
#include <limits>

JsonReader::JsonReader()
{
    _cur = _end = _begin = 0;
}
JsonReader::~JsonReader()
{
}
const std::string& JsonReader::error_message()
{
    return _error;
}
void JsonReader::error(const char* msg)
{
    int line, column;
    current_position(line, column);

    std::stringstream ss; ss << "(Line: " << line << ", Column: " << column << ") Error: " << msg;
    _error = ss.str();
}
void JsonReader::current_position(int& line, int& column)
{
    line = 0; column = 0;

    const char* current = _begin;
    const char* line_start = _begin;
    while (current < _cur)
    {
        if (*current == '\r')
        {
            if (*(current + 1) == '\n')
                ++current;

            line_start = current;
            ++line;
        }
        else if (*current == '\n')
        {
            line_start = current;
            ++line;
        }
        ++current;
    }
    ++line;
    column = (int)(_cur - line_start) - 1;
}
bool JsonReader::read(const char* doc, int64_t length, JsonObject& root)
{
    _cur = _begin = doc;
    _end = doc + length;

    skip_spaces();
    if (*_cur == '{')
        return parse_object(root);

    // Assume root is an object
    root.set_empty_object();

    std::string name;
    while (1)
    {
        skip_spaces();
        if (_cur == _end)
            break;


        name = "";
        if (!parse_string(name))
        {
            error("Failed to parse string");
            return false;
        }

        skip_spaces();
        if (*_cur != '=' && *_cur != ':')
        {
            error("Expected '=' or ':'");
            return false;
        }
        _cur++;

        JsonObject& elem = root[name.c_str()];
        if (!parse_value(elem))
        {
            return false; // Failed to parse value
        }

        skip_spaces();

        char c = *_cur;
        if (c == ',') // Separator between elements (Optional)
        {
            _cur++;
            continue;
        }
    }

    return true;
}
bool JsonReader::read_file(const std::string& file_name, JsonObject& root)
{
    // and then load any settings from settings file
    std::ifstream f(file_name, std::ifstream::in);
    if (!f.is_open())
    {
        std::stringstream ss;
        ss << "Failed to open file '" << file_name << "'";
        _error = ss.str();
        return false;
    }

    bool res = true;

    std::string buffer;
    f.seekg(0, std::ios::end);
    buffer.resize((size_t)f.tellg());
    f.seekg(0, std::ios::beg);
    f.read(&buffer[0], buffer.size());
    f.close();

    if (buffer.size() == 0)
    {
        // Empty file, just set root as empty object
        root.set_empty_object();
        res = true;
    }
    else
        res = read(buffer.c_str(), buffer.size(), root);

    return res;
}

void JsonReader::skip_spaces()
{
    while (_cur != _end)
    {
        char c = *_cur;
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
        {
            ++_cur;
        }
        else
            break;
    }
}

bool JsonReader::parse_string(std::string& str)
{
    // Typically a string is surrounded by quotes but we also support strings without quotes
    //	a string without quotes is considered to end at the first whitespace character (E.g. ' ' or '\t')

    // Count length first
    const char* str_end = _cur;
    bool quotes = (*_cur == '"'); // Keep track if this string is surrounded by quotes
    if (quotes)
        ++str_end; // Skip starting "
    while (str_end != _end)
    {
        char c = *str_end++;
        if (c == '\\')
            str_end++; // Skip checking next character	
        else if ((quotes && (c == '"')) || (!quotes && (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '=' || c == ':')))
            break;
    }
    if (quotes)
    {
        _cur += 1; // Skip starting "
    }
    str_end -= 1; // Skip trailing " or any trailing whitespace

    while (_cur != str_end)
    {
        char c = *_cur++;
        if (c == '\\')
        {
            char esc = *_cur++;
            switch (esc)
            {
            case '"':
                c = '"';
                break;
            case '/':
                c = '/';
                break;
            case '\\':
                c = '\\';
                break;
            case '\n':
                c = '\n';
                break;
            case '\r':
                c = '\r';
                break;
            case '\t':
                c = '\t';
                break;
            default:
                c = esc;
            };
        }

        str += c;
    }
    if (quotes)
        _cur++; // Trailing "

    return true;
}

bool JsonReader::parse_object(JsonObject& value)
{
    value.set_empty_object();

    _cur++; // Skip '{'
    skip_spaces();
    if (*_cur == '}') // Empty object
    {
        _cur++;
        return true;
    }

    std::string name;
    while (1)
    {
        skip_spaces();

        name = "";
        if (!parse_string(name))
        {
            error("Failed to parse string");
            break; // Failed to parse string
        }

        skip_spaces();
        if (*_cur != '=' && *_cur != ':')
        {
            error("Expected '=' or ':'");
            return false;
        }
        _cur++;

        JsonObject& elem = value[name.c_str()];
        if (!parse_value(elem))
            break; // Failed to parse value

        skip_spaces();

        char c = *_cur;
        if (c == ',') // Separator between elements (Optional)
        {
            _cur++;
            continue;
        }
        if (c == '}') // End of object
        {
            _cur++;
            return true;
        }
    }

    return false;
}
bool JsonReader::parse_array(JsonObject& value)
{
    value.set_empty_array();

    _cur++; // Skip '['
    skip_spaces();
    if (*_cur == ']')
    {
        _cur++;
        return true;
    }
    while (1)
    {
        JsonObject& elem = value.append();

        if (!parse_value(elem))
            return false;

        skip_spaces();

        char c = *_cur;
        if (c == ',') // Separator between elements (Optional)
        {
            _cur++;
            continue;
        }
        if (c == ']') // End of array
        {
            _cur++;
            break;
        }
    }
    return true;
}


bool JsonReader::parse_double(JsonObject& value)
{
    char* number_end;
    double number = std::strtod(_cur, &number_end);
    value.set_double(number);
    _cur = number_end;
    return true;
}

bool JsonReader::parse_number(JsonObject& value)
{
    bool integer = true; // Number is either integer or float
    for (const char* c = _cur; c != _end; ++c)
    {
        if ((*c >= '0' && *c <= '9') || ((*c == '-' || *c == '+') && (c == _cur))) // Allow a negative sign at the start for integers
            continue;
        else if (*c == '.' || *c == 'e' || *c == 'E' || *c == '+')
        {
            integer = false;
            break;
        }
        else
            break;
    }
    if (!integer)
        return parse_double(value);


    bool negative = (*_cur == '-');
    if (negative)
        _cur++;

    uint64_t number = 0;
    while (_cur != _end)
    {
        if (*_cur >= '0' && *_cur <= '9')
        {
            uint32_t digit = *_cur - '0';
            number = number * 10 + digit;
            _cur++;
        }
        else
            break;
    }
    if (negative)
    {
        value.set_int(-int64_t(number));
    }
    else if (number <= uint64_t(std::numeric_limits<int64_t>::max()))
    {
        value.set_int(int64_t(number));
    }
    else
    {
        value.set_uint(number);
    }

    return true;
}

bool JsonReader::parse_value(JsonObject& value)
{
    skip_spaces();
    bool b = true;
    char c = *_cur;
    switch (c)
    {
    case '{':
        b = parse_object(value);
        break;
    case '[':
        b = parse_array(value);
        break;
    case '"':
    {
        std::string str;
        b = parse_string(str);
        if (b)
            value.set_string(str.c_str());
        else
            error("Failed to parse string");
    }
    break;
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
    case '-':
        b = parse_number(value);
        break;
    case 't': // true
        if (*(++_cur) != 'r' || *(++_cur) != 'u' || *(++_cur) != 'e')
        {
            error("Expected \"true\"");
            return false;
        }
        ++_cur;
        value.set_bool(true);
        break;
    case 'f': // false
        if (*(++_cur) != 'a' || *(++_cur) != 'l' || *(++_cur) != 's' || *(++_cur) != 'e')
        {
            error("Expected \"false\"");
            return false;
        }
        ++_cur;
        value.set_bool(false);
        break;
    case 'n': // null
        if (*(++_cur) != 'u' || *(++_cur) != 'l' || *(++_cur) != 'l')
        {
            error("Expected \"null\"");
            return false;
        }
        ++_cur;
        value.set_null();
        break;
    default:
        b = false;
    };
    return b;
}

JsonWriter::JsonWriter() : _ilevel(0)
{
}
JsonWriter::~JsonWriter()
{
}
void JsonWriter::write(const JsonObject& root, std::stringstream& out, bool format)
{
    _format = format;
    write_value(root, out);
    out << "\n";
}
bool JsonWriter::write_file(const JsonObject& root, const std::string& file_name, bool format)
{
    _format = format;

    std::ofstream f(file_name, std::fstream::out);
    if (!f.is_open())
    {
        return false;
    }
    write_value(root, f);
    f << "\n"; // Trailing new-line

    return true;
}
void JsonWriter::write_value(const JsonObject& node, std::ostream& out)
{
    switch (node.type())
    {
    case JsonObject::NULL_VALUE:
        out << "null";
        break;
    case JsonObject::BOOL:
        if (node.as_bool())
            out << "true";
        else
            out << "false";
        break;
    case JsonObject::INTEGER:
        out << node.as_int64();
        break;
    case JsonObject::UINTEGER:
        out << node.as_uint64();
        break;
    case JsonObject::FLOAT:
        out << node.as_double();
        break;
    case JsonObject::STRING:
        out << "\"";
        write_string(node.as_string(), out);
        out << "\"";
        break;
    case JsonObject::ARRAY:
    {
        out << "[";

        _ilevel++;
        size_t size = node.size();
        for (size_t i = 0; i < size; ++i)
        {
            if (i != 0)
                out << ",";

            if (_format)
            {
                out << "\n";
                write_tabs(_ilevel, out);
            }
            write_value(node[(int)i], out);
        }
        if (_format)
            out << "\n";
        _ilevel--;
        if (_format)
            write_tabs(_ilevel, out);
        out << "]";
    }
    break;
    case JsonObject::OBJECT:
    {
        out << "{";
        _ilevel++;
        JsonObject::ConstIterator it, end;
        it = node.begin(); end = node.end();
        for (; it != end; ++it)
        {
            if (it != node.begin())
                out << ",";

            if (_format)
            {
                out << "\n";
                write_tabs(_ilevel, out);
            }
            out << "\"";
            write_string(it->first, out);
            out << "\": ";
            write_value(it->second, out);
        }
        if (_format)
            out << "\n";
        _ilevel--;
        if (_format)
            write_tabs(_ilevel, out);
        out << "}";
    }
    break;
    };
}


void JsonWriter::write_tabs(int ilevel, std::ostream& out)
{
    for (int i = 0; i < ilevel; ++i)
    {
        out << "\t";
    }
}

void JsonWriter::write_string(const std::string& str, std::ostream& out, bool quotes)
{
    // We need to escape any special characters before writing them to the JSON doc
    for (size_t i = 0; i < str.size(); ++i)
    {
	char c = str[i];

        switch (c)
        {
        case '\\':
            out << "\\\\";
            break;
        case '\"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        case ' ':
            if (!quotes) // If string isn't supposed to be surrouneded by quotes, we escape spaces
                out << "\\ ";
            else
                out << " ";
            break;
        default:
            out << c;
            break;

        }
    }
}


