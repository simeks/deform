#pragma once

#include <string>

#include <stk/math/float3.h>
#include <stk/math/matrix3x3f.h>

#include <stk/cuda/cuda.h>

class AffineTransform
{
public:
    AffineTransform();
    AffineTransform(const Matrix3x3f& matrix, const float3& offset);
    ~AffineTransform();

    CUDA_HOST_DEVICE inline float3 transform_point(const float3& pt) const
    {
        // Used by cuda::DisplacementField
        return _matrix * pt + _offset;
    }

    // Returns the resulting displacement vector at the given point
    inline float3 displacement(const float3& pt) const
    {
        return transform_point(pt) - pt;
    }

    const Matrix3x3f& matrix() const;
    const float3& offset() const;

private:
    Matrix3x3f _matrix;
    float3 _offset;
};

/*!
 * \brief Parses an affine transform file.
 * 
 * Supports:
 *  * Insight Transform File V1.0 
 */
AffineTransform parse_affine_transform_file(
    const std::string& filename
);
