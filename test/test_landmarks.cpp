#include "catch.hpp"

#include <deform_lib/registration/landmarks.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace Catch;

TEST_CASE("parse_landmarks")
{
    SECTION("test_parsing")
    {
        const char* landmarks_str =
                "point\n"
                "2  \n"
                "1.0\t3e-5 4.2\n"
                ".0   1  2\n";

        const std::string filename {"points.pts"};
        std::ofstream f(filename);
        assert(f.is_open());

        f.write(landmarks_str, std::strlen(landmarks_str));
        f.close();

        std::vector<float3> landmarks = parse_landmarks_file(filename);

        CHECK(2 == landmarks.size()) ;
        CHECK(1.0f  == landmarks[0].x);
        CHECK(3e-5f == landmarks[0].y);
        CHECK(4.2f  == landmarks[0].z);
        CHECK(0.0f  == landmarks[1].x);
        CHECK(1.0f  == landmarks[1].y);
        CHECK(2.0f  == landmarks[1].z);
    }

    SECTION("test_unexisting_file")
    {
        const std::string filename {"non_existing_file.pts"};
        std::vector<float3> landmarks;

        CHECK_THROWS_WITH(landmarks = parse_landmarks_file(filename),
                          Contains("Failed to open"));
    }

    SECTION("test_invalid_file_1")
    {
        const char* landmarks_str =
                "index\n"
                "2\n"
                "1 2 3\n"
                "4 5 6\n";

        const std::string filename {"points.pts"};
        std::ofstream f(filename);
        assert(f.is_open());

        f.write(landmarks_str, std::strlen(landmarks_str));
        f.close();

        std::vector<float3> landmarks;

        CHECK_THROWS_WITH(landmarks = parse_landmarks_file(filename),
                          Contains("expected \"point\""));
    }

    SECTION("test_invalid_file_2")
    {
        const char* landmarks_str =
                "point\n"
                "1 2 3\n"
                "4 5 6\n";

        const std::string filename {"points.pts"};
        std::ofstream f(filename);
        assert(f.is_open());

        f.write(landmarks_str, std::strlen(landmarks_str));
        f.close();

        std::vector<float3> landmarks;

        CHECK_THROWS_WITH(landmarks = parse_landmarks_file(filename),
                          Contains("expected number of points"));
    }

    SECTION("test_invalid_file_3")
    {
        const char* landmarks_str =
                "point\n"
                "2\n"
                "1 2 3\n"
                "4 5\n";

        const std::string filename {"points.pts"};
        std::ofstream f(filename);
        assert(f.is_open());

        f.write(landmarks_str, std::strlen(landmarks_str));
        f.close();

        std::vector<float3> landmarks;

        CHECK_THROWS_WITH(landmarks = parse_landmarks_file(filename),
                          Contains("Wrong number of components for point 2"));
    }

    SECTION("test_invalid_file_4")
    {
        const char* landmarks_str =
                "point\n"
                "3\n"
                "1 2 3\n"
                "4 5 6\n";

        const std::string filename {"points.pts"};
        std::ofstream f(filename);
        assert(f.is_open());

        f.write(landmarks_str, std::strlen(landmarks_str));
        f.close();

        std::vector<float3> landmarks;

        CHECK_THROWS_WITH(landmarks = parse_landmarks_file(filename),
                          Contains("Wrong number of components for point 3"));
    }
}

