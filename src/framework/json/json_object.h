#pragma once

#include <vector>
#include <map>
#include <sstream>

class JsonObjectIterator;
class JsonObjectConstIterator;

class JsonObject
{
public:
    enum ValueType
    {
        NULL_VALUE,
        INTEGER,
        UINTEGER,
        FLOAT,
        BOOL,
        STRING,
        ARRAY,
        OBJECT
    };

    typedef std::map<std::string, JsonObject> ValueMap;
    typedef JsonObjectIterator Iterator;
    typedef JsonObjectConstIterator ConstIterator;


public:
    JsonObject();
    JsonObject(const JsonObject& source);
    ~JsonObject();

    ValueType type() const;

    bool is_null() const;
    bool is_int() const;
    bool is_uint() const;
    bool is_float() const;
    bool is_bool() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;

    /// @return True if value is a number, meaning either an integer, unsigned integer or float
    bool is_number() const;

    /// @brief Returns the value of this object as an integer.
    int as_int() const;

    /// @brief Returns the value of this object as an 64bit integer.
    int64_t as_int64() const;

    /// @brief Returns the value of this object as an unsigned integer.
    uint32_t as_uint() const;

    /// @brief Returns the value of this object as an 64bit unsigned integer.
    uint64_t as_uint64() const;

    /// @brief Returns the value of this object as a floating-point value.
    float as_float() const;

    /// @brief Returns the value of this object as a double precision float.
    double as_double() const;

    /// @brief Returns the value of this object as a boolean.
    bool as_bool() const;

    /// @brief Returns the value of this object as a string.
    const std::string& as_string() const;

    /// @brief Returns the value of this object as an array.
    const std::vector<JsonObject>& as_array() const;

    /// @brief Returns the value of this object as a value map.
    const ValueMap& as_map() const;

    /// @brief Returns the size of this value, the number of sub elements
    /// @return Number of sub elements if either an array or an object. 
    ///			If the value is a single element type this returns 1, and if
    ///			the value is NULL it returns 0.
    size_t size() const;


    /// @brief Sets the object value to null, releasing any previous value.
    void set_null();

    /// @brief Sets the objects value to the specified integer.
    void set_int(int i);

    /// @brief Sets the objects value to the specified integer.
    void set_int(int64_t i);

    /// @brief Sets the objects value to the specified unsigned integer.
    void set_uint(uint32_t i);

    /// @brief Sets the objects value to the specified unsigned integer.
    void set_uint(uint64_t i);

    /// @brief Sets the objects value to the specified float.
    void set_float(float f);

    /// @brief Sets the objects value to the specified float.
    void set_double(double d);

    /// @brief Sets the objects value to the specified boolean.
    void set_bool(bool b);

    /// @brief Sets the objects value to the specified string.
    void set_string(const std::string& s);

    /// @brief Sets this value to an empty array
    void set_empty_array();

    /// @brief Sets this value to an empty object
    void set_empty_object();

    /// @brief Appends a value to the array
    /// @remark Assumes that this JsonObject is an array
    JsonObject& append();

    /// @brief Returns an iterator for the beginning of all object elements 
    /// @remark This only works if the value is of the type OBJECT
    Iterator begin();

    /// @brief Returns an iterator for the beginning of all object elements 
    /// @remark This only works if the value is of the type OBJECT
    ConstIterator begin() const;

    /// @brief Returns the end iterator for the object elements
    /// @remark This only works if the value is of the type OBJECT
    Iterator end();

    /// @brief Returns the end iterator for the object elements
    /// @remark This only works if the value is of the type OBJECT
    ConstIterator end() const;

    JsonObject& operator=(const JsonObject& source);

    JsonObject& operator[](const char* key);
    const JsonObject& operator[](const char* key) const;

    JsonObject& operator[](int index);
    const JsonObject& operator[](int index) const;

private:

    union Value
    {
        int64_t i;
        uint64_t u;
        double d;
        bool b;
        std::string* s;
        std::vector<JsonObject>* a;
        ValueMap* o;
    };

    ValueType _type;
    Value _value;

};

class JsonObjectIterator : public JsonObject::ValueMap::iterator
{
public:
    JsonObjectIterator() : JsonObject::ValueMap::iterator() {}
    JsonObjectIterator(JsonObject::ValueMap::iterator iter) :
        JsonObject::ValueMap::iterator(iter) { }
};

class JsonObjectConstIterator : public JsonObject::ValueMap::const_iterator
{
public:
    JsonObjectConstIterator() : JsonObject::ValueMap::const_iterator() {}
    JsonObjectConstIterator(JsonObject::ValueMap::const_iterator iter) :
        JsonObject::ValueMap::const_iterator(iter) { }
};


