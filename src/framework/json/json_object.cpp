#include "json_object.h"

#include "debug/assert.h"

JsonObject::JsonObject()
    : _type(NULL_VALUE)
{
}
JsonObject::JsonObject(const JsonObject& source)
    : _type(source._type)
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        _value = source._value;
        break;
    case STRING:
        _value.s = new std::string(*source._value.s);
        break;
    case ARRAY:
        _value.a = new std::vector<JsonObject>(*source._value.a);
        break;
    case OBJECT:
        _value.o = new ValueMap(*source._value.o);
        break;
    };
}
JsonObject::~JsonObject()
{
    set_null();
}

JsonObject::ValueType JsonObject::type() const
{
    return _type;
}
bool JsonObject::is_null() const
{
    return (_type == NULL_VALUE);
}
bool JsonObject::is_int() const
{
    return (_type == INTEGER);
}
bool JsonObject::is_uint() const
{
    return (_type == UINTEGER);
}
bool JsonObject::is_float() const
{
    return (_type == FLOAT);
}
bool JsonObject::is_bool() const
{
    return (_type == BOOL);
}
bool JsonObject::is_string() const
{
    return (_type == STRING);
}
bool JsonObject::is_array() const
{
    return (_type == ARRAY);
}
bool JsonObject::is_object() const
{
    return (_type == OBJECT);
}
bool JsonObject::is_number() const
{
    return (_type == INTEGER || _type == UINTEGER || _type == FLOAT);
}

int JsonObject::as_int() const
{
    switch (_type)
    {
    case NULL_VALUE:
        return 0;
    case INTEGER:
        return int(_value.i);
    case UINTEGER:
        return int(_value.u);
    case FLOAT:
        return int(_value.d);
    case BOOL:
        return (_value.b ? 1 : 0);
    case STRING:
    case ARRAY:
    case OBJECT:
        assert(false);
    };
    return 0;
}
int64_t JsonObject::as_int64() const
{
    switch (_type)
    {
    case NULL_VALUE:
        return 0;
    case INTEGER:
        return _value.i;
    case UINTEGER:
        return int64_t(_value.u);
    case FLOAT:
        return int64_t(_value.d);
    case BOOL:
        return (_value.b ? 1 : 0);
    case STRING:
    case ARRAY:
    case OBJECT:
        assert(false);
    };
    return 0;
}
uint32_t JsonObject::as_uint() const
{
    switch (_type)
    {
    case NULL_VALUE:
        return 0;
    case INTEGER:
        return uint32_t(_value.i);
    case UINTEGER:
        return uint32_t(_value.u);
    case FLOAT:
        return int(_value.d);
    case BOOL:
        return (_value.b ? 1 : 0);
    case STRING:
    case ARRAY:
    case OBJECT:
        assert(false);
    };
    return 0;
}
uint64_t JsonObject::as_uint64() const
{
    switch (_type)
    {
    case NULL_VALUE:
        return 0;
    case INTEGER:
        return uint64_t(_value.i);
    case UINTEGER:
        return uint64_t(_value.u);
    case FLOAT:
        return uint64_t(_value.d);
    case BOOL:
        return (_value.b ? 1 : 0);
    case STRING:
    case ARRAY:
    case OBJECT:
        assert(false);
    };
    return 0;
}
float JsonObject::as_float() const
{
    switch (_type)
    {
    case NULL_VALUE:
        return 0.0f;
    case INTEGER:
        return float(_value.i);
    case UINTEGER:
        return float(_value.u);
    case FLOAT:
        return float(_value.d);
    case BOOL:
        return (_value.b ? 1.0f : 0.0f);
    case STRING:
    case ARRAY:
    case OBJECT:
        assert(false);
    };
    return 0.0f;
}
double JsonObject::as_double() const
{
    switch (_type)
    {
    case NULL_VALUE:
        return 0.0;
    case INTEGER:
        return double(_value.i);
    case UINTEGER:
        return double(_value.u);
    case FLOAT:
        return _value.d;
    case BOOL:
        return (_value.b ? 1.0 : 0.0);
    case STRING:
    case ARRAY:
    case OBJECT:
        assert(false);
    };
    return 0.0;
}
bool JsonObject::as_bool() const
{
    switch (_type)
    {
    case NULL_VALUE:
        return false;
    case INTEGER:
        return _value.i != 0;
    case UINTEGER:
        return _value.u != 0;
    case FLOAT:
        return _value.d != 0.0;
    case BOOL:
        return _value.b;
    case STRING:
    case ARRAY:
    case OBJECT:
        assert(false);
    };
    return false;
}
const std::string& JsonObject::as_string() const
{
    assert(_type == STRING);
    return *_value.s;
}
const std::vector<JsonObject>& JsonObject::as_array() const
{
    assert(_type == ARRAY);
    return *_value.a;
}
const JsonObject::ValueMap& JsonObject::as_map() const
{
    assert(_type == OBJECT);
    return *_value.o;
}
size_t JsonObject::size() const
{
    switch (_type)
    {
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        return 1;
    case STRING:
        return _value.s->size();
    case ARRAY:
        return _value.a->size();
    case OBJECT:
        return _value.o->size();
    case NULL_VALUE:
        return 0;
    };
    return 0;
}

void JsonObject::set_null()
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case STRING:
        delete _value.s;
        break;
    case ARRAY:
        delete _value.a;
        break;
    case OBJECT:
        delete _value.o;
        break;
    };
    _type = NULL_VALUE;
    _value.i = 0;
}
void JsonObject::set_int(int i)
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case STRING:
    case ARRAY:
    case OBJECT:
        set_null();
        break;
    };
    _type = INTEGER;
    _value.i = i;
}
void JsonObject::set_int(int64_t i)
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case STRING:
    case ARRAY:
    case OBJECT:
        set_null();
        break;
    };
    _type = INTEGER;
    _value.i = i;
}
void JsonObject::set_uint(uint32_t i)
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case STRING:
    case ARRAY:
    case OBJECT:
        set_null();
        break;
    };
    _type = UINTEGER;
    _value.u = i;
}
void JsonObject::set_uint(uint64_t i)
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case STRING:
    case ARRAY:
    case OBJECT:
        set_null();
        break;
    };
    _type = UINTEGER;
    _value.u = i;
}
void JsonObject::set_float(float f)
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case STRING:
    case ARRAY:
    case OBJECT:
        set_null();
        break;
    };
    _type = FLOAT;
    _value.d = f;
}
void JsonObject::set_double(double d)
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case STRING:
    case ARRAY:
    case OBJECT:
        set_null();
        break;
    };
    _type = FLOAT;
    _value.d = d;
}
void JsonObject::set_bool(bool b)
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case STRING:
    case ARRAY:
    case OBJECT:
        set_null();
        break;
    };
    _type = BOOL;
    _value.b = b;
}
void JsonObject::set_string(const std::string& s)
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case ARRAY:
    case OBJECT:
        set_null();
        break;
    case STRING:
        _value.s->clear();
        *_value.s = s;
        return;
    };

    _type = STRING;
    _value.s = new std::string(s);
    *_value.s = s;
}
void JsonObject::set_empty_array()
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case STRING:
    case OBJECT:
        set_null();
        break;
    case ARRAY:
        _value.a->clear();
        return;
    };

    _type = ARRAY;
    _value.a = new std::vector<JsonObject>;

}
void JsonObject::set_empty_object()
{
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        break;
    case STRING:
    case ARRAY:
        set_null();
        break;
    case OBJECT:
        _value.o->clear();
        return;
    };

    _type = OBJECT;
    _value.o = new ValueMap;
}
JsonObject& JsonObject::append()
{
    assert(_type == ARRAY);
    _value.a->push_back(JsonObject());
    return _value.a->back();
}

JsonObject::Iterator JsonObject::begin()
{
    assert(_type == OBJECT);
    return Iterator(_value.o->begin());
}
JsonObject::ConstIterator JsonObject::begin() const
{
    assert(_type == OBJECT);
    return ConstIterator(_value.o->begin());
}
JsonObject::Iterator JsonObject::end()
{
    assert(_type == OBJECT);
    return Iterator(_value.o->end());
}
JsonObject::ConstIterator JsonObject::end() const
{
    assert(_type == OBJECT);
    return ConstIterator(_value.o->end());
}

JsonObject& JsonObject::operator=(const JsonObject& source)
{
    set_null();

    _type = source._type;
    switch (_type)
    {
    case NULL_VALUE:
    case INTEGER:
    case UINTEGER:
    case FLOAT:
    case BOOL:
        _value = source._value;
        break;
    case STRING:
        _value.s = new std::string;
        *_value.s = *source._value.s;
        break;
    case ARRAY:
        _value.a = new std::vector<JsonObject>;
        *_value.a = *source._value.a;
        break;
    case OBJECT:
        _value.o = new ValueMap;
        *_value.o = *source._value.o;
        break;
    };

    return *this;
}
JsonObject& JsonObject::operator[](const char* key)
{
    assert(_type == OBJECT);
    JsonObject& ret = (*_value.o)[std::string(key)];

    return ret;
}
const JsonObject& JsonObject::operator[](const char* key) const
{
    assert(_type == OBJECT);
    return (*_value.o)[std::string(key)];
}

JsonObject& JsonObject::operator[](int index)
{
    assert(_type == ARRAY);
    return (*_value.a)[index];
}
const JsonObject& JsonObject::operator[](int index) const
{
    assert(_type == ARRAY);
    return (*_value.a)[index];
}

