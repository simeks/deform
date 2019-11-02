#include "affine.h"

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


AffineTransform parse_affine_transform_file(
    const std::string& filename
)
{
    // TODO: Cleanup, should probably be less strict (ordering of keys, etc)

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

    auto split = [](const std::string& line) {
        size_t p = line.find(':');
        if (p != std::string::npos) {
            return std::pair<std::string, std::string>(
                trim_string(line.substr(0, p)),
                trim_string(line.substr(p+1, std::string::npos))
            );
        }
        return std::pair<std::string, std::string>(
            "",""
        );
    };

    // Transform: AffineTransform_double_3_3
    std::getline(f, line);
    auto p = split(line);

    if (p.first != "Transform") {
        throw ValidationError("Invalid file format");
    }

    if (p.second != "AffineTransform_double_3_3") {
        throw ValidationError("Invalid transform name, only 'AffineTransform_double_3_3' is supported.");
    }

    // Parameters: <x>*12
    std::getline(f, line);
    p = split(line);
    if (p.first != "Parameters") {
        throw ValidationError("Invalid file format");
    }

    Matrix3x3f matrix = Matrix3x3f::Identity;
    float* pmatrix = &matrix._rows[0].x;
    std::istringstream s_p(p.second);

    for (int i = 0; i < 9; ++i) {
        s_p >> pmatrix[i];
    }

    float3 translation;
    s_p >> translation.x;
    s_p >> translation.y;
    s_p >> translation.z;

    // FixedParameters: <x> <y> <z>
    std::getline(f, line);
    p = split(line);
    if (p.first != "FixedParameters") {
        throw ValidationError("Invalid file format");
    }

    std::istringstream s_fp(p.second);

    float3 fixed_parameters;
    s_fp >> fixed_parameters.x;
    s_fp >> fixed_parameters.y;
    s_fp >> fixed_parameters.z;

    float3 center = fixed_parameters;
    float3 offset = translation + center - matrix * center;
    
    return AffineTransform(matrix, offset);
}
