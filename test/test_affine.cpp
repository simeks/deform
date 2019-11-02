#include "catch.hpp"

#include <deform_lib/registration/affine.h>
#include <stk/math/types.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace Catch;

TEST_CASE("parse_affine")
{
    SECTION("test_parsing")
    {
        const char* affine_str =
            "#Insight Transform File V1.0\n"
            "#Transform 0\n"
            "Transform: AffineTransform_double_3_3\n"
            "Parameters: 1 2 3 4 5 6 7 8 9 10 11 12\n"
            "FixedParameters: 13 14 15\n";

        const std::string filename {"affine.txt"};
        std::ofstream f(filename);
        assert(f.is_open());

        f.write(affine_str, std::strlen(affine_str));
        f.close();

        AffineTransform transform = parse_affine_transform_file(filename);

        float3 pt1 = transform.transform_point(float3{1,2,3});
        float3 pt2 = transform.transform_point(float3{-5.1,-7.2,-8.3});

        // Expected values computed by SimpleITK
        
        CHECK(-49.0f == pt1.x);
        CHECK(-155.0f == pt1.y);
        CHECK(-261.0f == pt1.z);

        CHECK(-107.4f == pt2.x);
        CHECK(-293.2f == pt2.y);
        CHECK(-479.0f == pt2.z);
    }

    SECTION("test_unexisting_file")
    {
        const std::string filename {"non_existing_file.pts"};

        CHECK_THROWS_WITH(parse_affine_transform_file(filename),
            Contains("Failed to open"));
    }

    SECTION("test_invalid_file_1")
    {
        const char* affine_str =
            "#Transform 0\n"
            "Transform: AffineTransform_double_3_3\n"
            "Parameters: 1 2 3 4 5 6 7 8 9 10 11 12\n"
            "FixedParameters: 13 14 15\n";

        const std::string filename {"affine.txt"};
        std::ofstream f(filename);
        assert(f.is_open());

        f.write(affine_str, std::strlen(affine_str));
        f.close();

        CHECK_THROWS_WITH(parse_affine_transform_file(filename),
            Contains("Unsupported transform file format"));
    }

    SECTION("test_invalid_file_2")
    {
        const char* affine_str =
            "#Insight Transform File V1.0\n";

        const std::string filename {"affine.txt"};
        std::ofstream f(filename);
        assert(f.is_open());

        f.write(affine_str, std::strlen(affine_str));
        f.close();

        CHECK_THROWS_WITH(parse_affine_transform_file(filename),
            Contains("Invalid file format"));
    }

    SECTION("test_invalid_file_2")
    {
        const char* affine_str =
            "#Insight Transform File V1.0\n"
            "#Transform 0\n"
            "Transform: AffineTransform_double_1_1\n";

        const std::string filename {"affine.txt"};
        std::ofstream f(filename);
        assert(f.is_open());

        f.write(affine_str, std::strlen(affine_str));
        f.close();

        CHECK_THROWS_WITH(parse_affine_transform_file(filename),
            Contains("Invalid transform name"));
    }
}
