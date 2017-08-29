#pragma once

#ifdef DF_ENABLE_CUDA
// float2, float3, etc...
#include <vector_types.h>

#else

#include <stdint.h>

// Defines some types otherwise defined by CUDA SDK.
// However, not the same guarantees on alignmnet.

struct uchar2
{
    uint8_t x, y;
};

struct uchar3
{
    uint8_t x, y, z;
};

struct uchar4
{
    uint8_t x, y, z, w;
};

struct int2
{
    int x, y;
};

struct int3
{
    int x, y, z;
};

struct int4
{
    int x, y, z, w;
};

struct float2
{
    float x, y;
};

struct float3
{
    float x, y, z;
};

struct float4
{
    float x, y, z, w;
};

struct double2
{
    double x, y;
};

struct double3
{
    double x, y, z;
};

struct double4
{
    double x, y, z, w;
};

#endif