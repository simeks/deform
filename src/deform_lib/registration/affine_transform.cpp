#include "affine_transform.h"

#include <deform_lib/registration/settings.h>
#include <stk/math/float3.h>
#include <stk/math/matrix3x3f.h>

#include <fstream>

namespace {
    std::string trim_string(const std::string& str)
    {
        size_t i = 0, j = str.size()-1; 

        while (i < j && isspace(str[i])) i++;
        while (j > i && isspace(str[j])) j--;

        return std::string(str, i, j+1);
    }
}

AffineTransform::AffineTransform() :
    _matrix(Matrix3x3f::Identity),
    _offset{0, 0, 0}
{
}
AffineTransform::AffineTransform(const Matrix3x3f& matrix, const float3& offset) :
    _matrix(matrix),
    _offset(offset)
{
}
AffineTransform::~AffineTransform()
{
}

const Matrix3x3f& AffineTransform::matrix() const
{
    return _matrix;
}
const float3& AffineTransform::offset() const
{
    return _offset;
}


AffineTransform parse_affine_transform_file(const std::string& filename)
{
    std::string line;

    std::ifstream f {filename, std::ifstream::in};
    if (!f.is_open()) {
        throw ValidationError("Failed to open file '" + filename + "'");
    }

    std::getline(f, line);
    if (line != "#Insight Transform File V1.0") {
        throw ValidationError("Unsupported transform file format");
    }

    // #Transform <i>
    std::getline(f, line);

    std::map<std::string, std::string> keyvalues;
    while (std::getline(f, line))
    {
        size_t p = line.find(':');
        if (p != std::string::npos) {
            std::string key = trim_string(line.substr(0, p));
            std::string value = trim_string(line.substr(p+1, std::string::npos));

            keyvalues[key] = value;
        }
    }

    if (keyvalues.find("Transform") != keyvalues.end()) {
        if (keyvalues["Transform"] != "AffineTransform_double_3_3") {
            throw ValidationError(
                "Invalid transform name, only 'AffineTransform_double_3_3' is supported."
            );
        }
    }
    else {
        throw ValidationError("Invalid file format: Missing \"Transform\"");
    }

    Matrix3x3f matrix = Matrix3x3f::Identity;
    float3 translation{0, 0, 0};

    if (keyvalues.find("Parameters") != keyvalues.end()) {
        // Parameters: <x>*12

        std::istringstream ss(keyvalues["Parameters"]);

        float* pmatrix = &matrix._rows[0].x;
        for (int i = 0; i < 9; ++i) {
            ss >> pmatrix[i];
        }

        ss >> translation.x;
        ss >> translation.y;
        ss >> translation.z;
    }
    else {
        throw ValidationError("Invalid file format: Missing \"Parameters\"");
    }

    float3 fixed_parameters{0, 0, 0};

    // FixedParameters are optional
    if (keyvalues.find("FixedParameters") != keyvalues.end()) {
        // FixedParameters: <x> <y> <z>

        std::istringstream ss(keyvalues["FixedParameters"]);
        ss >> fixed_parameters.x;
        ss >> fixed_parameters.y;
        ss >> fixed_parameters.z;
    }

    float3 center = fixed_parameters;
    float3 offset = translation + center - matrix * center;
    
    return AffineTransform(matrix, offset);
}
