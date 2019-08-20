#include "catch.hpp"

#include <deform_lib/registration/settings.h>

#include <cstring>
#include <fstream>

using namespace Catch;

namespace {
const char* sample_settings = R"(
pyramid_levels: 4
pyramid_stop_level: 2

block_size: [20, 24, 38]
block_energy_epsilon: 0.00000000009
max_iteration_count: 100
step_size: 10.5
regularization_weight: 0.5
regularization_scale: 1.1
regularization_exponent: 1.5

constraints_weight: 1234.1234

regularize_initial_displacement: true

levels:
    3:
        block_size: [9,9,9]
        block_energy_epsilon: 0.9
        max_iteration_count: 99
        step_size: 9.9
        regularization_weight: 9
        regularization_scale: 0.9
        regularization_exponent: 2.0

        constraints_weight: 999.999

image_slots:
  # water
  - resampler: gaussian
    normalize: true
    cost_function: none

  # fat
  - resampler: gaussian
    normalize: true
    cost_function: squared_distance

  # mask
  - resampler: gaussian
    normalize: true
    cost_function: ncc

  - resampler: gaussian
    normalize: false
    cost_function: none

  - resampler: gaussian
    normalize: false
    cost_function:
      - function: ssd
        weight: 0.4
      - function: ncc
        weight: 0.3

  - resampler: gaussian
    normalize: false
    cost_function: ncc

  - resampler: gaussian
    normalize: false
    cost_function: squared_distance

  - resampler: gaussian
    normalize: false
    cost_function: none
)";
const char* broken_sample_settings = R"(
pyramid_levels: 4
pyramid_stop_level: 2

block_size: [20, 24
": 10.5
regularization_weight: [0, 1, 2, 3]

constraints_weight: 1234.1234

image_slots:
  - resampler: gaussian
    normalize: true
    cost_function: none
  - resampler: gaussian
    normalize: true
    cost_function: squared_distance
  - resampler: gaussian
    normalize: true
    cost_function: ncc
  - resampler: gaussian
    normalize: false
    cost_function: none
  - resampler: gaussian
    normalize: false
    cost_function: squared_distance
  - resampler: gaussian
    normalize: false
    cost_function: ncc
  - resampler: gaussian
    normalize: false
    cost_function: squared_distance
  - resampler: gaussian
    normalize: false
    cost_function: none
)";
}

TEST_CASE("parse_registration_settings")
{
    SECTION("test_cost_function_single_component")
    {
        Settings settings;
        std::string settings_str =
                "image_slots:\n"
                "  - resampler: gaussian\n"
                "    normalize: true\n"
                "    cost_function: ncc\n";

        REQUIRE(parse_registration_settings(settings_str, settings));

        REQUIRE(settings.image_slots[0].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[0].normalize == true);
        REQUIRE(settings.image_slots[0].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_NCC);
        REQUIRE(settings.image_slots[0].cost_functions[0].weight == 1.0f);
    }
    SECTION("test_cost_function_multi_component_1")
    {
        Settings settings;
        std::string settings_str =
                "image_slots:\n"
                "  - resampler: gaussian\n"
                "    normalize: true\n"
                "    cost_function:\n"
                "      - function: ncc\n"
                "        weight: 0.3\n";

        REQUIRE(parse_registration_settings(settings_str, settings));

        REQUIRE(settings.image_slots[0].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[0].normalize == true);
        REQUIRE(settings.image_slots[0].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_NCC);
        REQUIRE(settings.image_slots[0].cost_functions[0].weight == 0.3f);
    }
    SECTION("test_ncc_window")
    {
        Settings settings;
        std::string  settings_str;

        // cube
        settings_str =
                "image_slots:\n"
                "  - cost_function:\n"
                "      - function: ncc\n"
                "        window: cube\n"
                "        radius: 3";

        REQUIRE(parse_registration_settings(settings_str, settings));

        REQUIRE(settings.image_slots[0].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_NCC);
        REQUIRE(settings.image_slots[0].cost_functions[0].parameters["window"] == "cube");
        REQUIRE(settings.image_slots[0].cost_functions[0].parameters["radius"] == "3");

        // sphere
        settings_str =
                "image_slots:\n"
                "  - cost_function:\n"
                "      - function: ncc\n"
                "        window: sphere\n"
                "        radius: 4";

        REQUIRE(parse_registration_settings(settings_str, settings));

        REQUIRE(settings.image_slots[0].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_NCC);
        REQUIRE(settings.image_slots[0].cost_functions[0].parameters["window"] == "sphere");
        REQUIRE(settings.image_slots[0].cost_functions[0].parameters["radius"] == "4");
    }
    SECTION("test_cost_function_multi_component_2")
    {
        Settings settings;
        std::string settings_str =
                "image_slots:\n"
                "  - resampler: gaussian\n"
                "    normalize: true\n"
                "    cost_function:\n"
                "      - function: ncc\n"
                "        weight: 0.5\n"
                "      - function: ssd\n"
                "        weight: 0.8\n";

        REQUIRE(parse_registration_settings(settings_str, settings));

        REQUIRE(settings.image_slots[0].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[0].normalize == true);
        REQUIRE(settings.image_slots[0].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_NCC);
        REQUIRE(settings.image_slots[0].cost_functions[0].weight == 0.5f);
        REQUIRE(settings.image_slots[0].cost_functions[1].function ==
                Settings::ImageSlot::CostFunction_SSD);
        REQUIRE(settings.image_slots[0].cost_functions[1].weight == 0.8f);
    }
    SECTION("test_cost_function_broken_1")
    {
        Settings settings;
        std::string settings_str =
                "image_slots:\n"
                "  - resampler: gaussian\n"
                "    normalize: true\n"
                "    cost_function:\n"
                "      - function: ncc\n"
                "      - weight: 0.8\n";

        REQUIRE(!parse_registration_settings(settings_str, settings));
    }
    SECTION("test_cost_function_broken_2")
    {
        Settings settings;
        std::string settings_str =
                "image_slots:\n"
                "  - resampler: gaussian\n"
                "    normalize: true\n"
                "    cost_function:\n"
                "      - function: crt\n"
                "        weight: 0.3\n";

        REQUIRE(!parse_registration_settings(settings_str, settings));
    }
    SECTION("test_cost_function_broken_3")
    {
        Settings settings;
        std::string settings_str =
                "image_slots:\n"
                "  - resampler: gaussian\n"
                "    normalize: true\n"
                "    cost_function:\n"
                "      - function: 0.3\n"
                "        weight: 0.3\n";

        REQUIRE(!parse_registration_settings(settings_str, settings));
    }
    SECTION("test_cost_function_broken_4")
    {
        Settings settings;
        std::string settings_str =
                "image_slots:\n"
                "  - resampler: foo\n"
                "    normalize: true\n"
                "    cost_function:\n"
                "      - function: ncc\n"
                "        weight: 0.3\n";

        REQUIRE(!parse_registration_settings(settings_str, settings));
    }
    SECTION("test_vectorial_step_size")
    {
        Settings settings;
        std::string settings_str =
                "step_size: [1.1, 2.2, 3.3]\n";

        REQUIRE(parse_registration_settings(settings_str, settings));
        for (int i = 0; i < settings.num_pyramid_levels; ++i) {
                REQUIRE(settings.levels[i].step_size.x == Approx(1.1f));
                REQUIRE(settings.levels[i].step_size.y == Approx(2.2f));
                REQUIRE(settings.levels[i].step_size.z == Approx(3.3f));
        }
    }
    SECTION("test_vectorial_step_size_broken")
    {
        Settings settings;

        std::string settings_str = "step_size: foo\n";
        CHECK(!parse_registration_settings(settings_str, settings));

        settings_str = "step_size: [1.0, 2.0]\n";
        CHECK(!parse_registration_settings(settings_str, settings));
    }
}

TEST_CASE("parse_registration_file", "")
{
    SECTION("test_file")
    {
        std::ofstream f("test_settings.yml");
        REQUIRE(f.is_open());

        f.write(sample_settings, strlen(sample_settings));
        f.close();

        Settings settings;
        REQUIRE(parse_registration_file("test_settings.yml", settings));

        REQUIRE(settings.pyramid_stop_level == 2);
        REQUIRE(settings.num_pyramid_levels == 4);
        REQUIRE(settings.regularize_initial_displacement == true);

        for (int i = 0; i < settings.num_pyramid_levels; ++i) {
            if (i == 3) {
                REQUIRE(settings.levels[i].block_size.x == 9);
                REQUIRE(settings.levels[i].block_size.y == 9);
                REQUIRE(settings.levels[i].block_size.z == 9);

                REQUIRE(settings.levels[i].block_energy_epsilon == Approx(0.9));
                REQUIRE(settings.levels[i].max_iteration_count == 99);
                REQUIRE(settings.levels[i].step_size.x == Approx(9.9f));
                REQUIRE(settings.levels[i].step_size.y == Approx(9.9f));
                REQUIRE(settings.levels[i].step_size.z == Approx(9.9f));

                REQUIRE(settings.levels[i].regularization_weight == Approx(9.0f));
                REQUIRE(settings.levels[i].regularization_scale == Approx(0.9f));
                REQUIRE(settings.levels[i].regularization_exponent == Approx(2.0f));
                REQUIRE(settings.levels[i].constraints_weight == Approx(999.999f));
            }
            else {
                REQUIRE(settings.levels[i].block_size.x == 20);
                REQUIRE(settings.levels[i].block_size.y == 24);
                REQUIRE(settings.levels[i].block_size.z == 38);

                REQUIRE(settings.levels[i].block_energy_epsilon == Approx(0.00000000009));
                REQUIRE(settings.levels[i].max_iteration_count == 100);
                REQUIRE(settings.levels[i].step_size.x == Approx(10.5f));
                REQUIRE(settings.levels[i].step_size.y == Approx(10.5f));
                REQUIRE(settings.levels[i].step_size.z == Approx(10.5f));

                REQUIRE(settings.levels[i].regularization_weight == Approx(0.5f));
                REQUIRE(settings.levels[i].regularization_scale == Approx(1.1f));
                REQUIRE(settings.levels[i].regularization_exponent == Approx(1.5f));
                REQUIRE(settings.levels[i].constraints_weight == Approx(1234.1234f));
            }
        }

        REQUIRE(settings.image_slots[0].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[0].normalize == true);
        REQUIRE(settings.image_slots[0].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_None);
        REQUIRE(settings.image_slots[0].cost_functions[0].weight ==
                1.0f);

        REQUIRE(settings.image_slots[1].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[1].normalize == true);
        REQUIRE(settings.image_slots[1].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_SSD);
        REQUIRE(settings.image_slots[1].cost_functions[0].weight ==
                1.0f);

        REQUIRE(settings.image_slots[2].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[2].normalize == true);
        REQUIRE(settings.image_slots[2].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_NCC);
        REQUIRE(settings.image_slots[2].cost_functions[0].weight ==
                1.0f);

        REQUIRE(settings.image_slots[3].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[3].normalize == false);
        REQUIRE(settings.image_slots[3].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_None);
        REQUIRE(settings.image_slots[3].cost_functions[0].weight ==
                1.0f);

        REQUIRE(settings.image_slots[4].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[4].normalize == false);
        REQUIRE(settings.image_slots[4].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_SSD);
        REQUIRE(settings.image_slots[4].cost_functions[0].weight ==
                0.4f);
        REQUIRE(settings.image_slots[4].cost_functions[1].function ==
                Settings::ImageSlot::CostFunction_NCC);
        REQUIRE(settings.image_slots[4].cost_functions[1].weight ==
                0.3f);

        REQUIRE(settings.image_slots[5].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[5].normalize == false);
        REQUIRE(settings.image_slots[5].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_NCC);
        REQUIRE(settings.image_slots[5].cost_functions[0].weight ==
                1.0f);

        REQUIRE(settings.image_slots[6].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[6].normalize == false);
        REQUIRE(settings.image_slots[6].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_SSD);
        REQUIRE(settings.image_slots[6].cost_functions[0].weight ==
                1.0f);

        REQUIRE(settings.image_slots[7].resample_method ==
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[7].normalize == false);
        REQUIRE(settings.image_slots[7].cost_functions[0].function ==
                Settings::ImageSlot::CostFunction_None);
        REQUIRE(settings.image_slots[7].cost_functions[0].weight ==
                1.0f);
    }
    SECTION("no_file")
    {
        Settings settings;
        REQUIRE(!parse_registration_file("no_file_here.yml", settings));
    }
    SECTION("broken_file")
    {
        std::ofstream f("broken_test_settings.yml");
        REQUIRE(f.is_open());

        f.write(broken_sample_settings, strlen(broken_sample_settings));
        f.close();

        Settings settings;
        REQUIRE(!parse_registration_file("broken_test_settings.yml", settings));
    }
}
