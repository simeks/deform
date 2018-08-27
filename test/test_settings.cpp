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
step_size: 10.5
regularization_weight: [0, 1, 2, 3]

constraints_weight: 1234.1234

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
    SECTION("test_regularization_weight_single_component")
    {
        Settings settings;
        std::string settings_str =
                "pyramid_levels: 8\n"
                "regularization_weight: 0.5\n";

        REQUIRE(parse_registration_settings(settings_str, settings));

        for (int i = 0; i < 8; ++i) {
            REQUIRE(settings.regularization_weights[i] == Approx(0.5f));
        }
    }
    SECTION("test_regularization_weight_multi_component")
    {
        Settings settings;
        std::string settings_str =
                "pyramid_levels: 8\n"
                "regularization_weight: [0, 1, 2, 3, 4, 5, 6, 7]\n";

        REQUIRE(parse_registration_settings(settings_str, settings));

        for (int i = 0; i < 8; ++i) {
            REQUIRE(settings.regularization_weights[i] == Approx(float(i)));
        }
    }
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
        REQUIRE(settings.step_size.x == Approx(1.1f));
        REQUIRE(settings.step_size.y == Approx(2.2f));
        REQUIRE(settings.step_size.z == Approx(3.3f));
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

        REQUIRE(settings.block_size.x == 20);
        REQUIRE(settings.block_size.y == 24);
        REQUIRE(settings.block_size.z == 38);
        
        REQUIRE(settings.block_energy_epsilon == Approx(0.00000000009));
        REQUIRE(settings.step_size.x == Approx(10.5f));
        REQUIRE(settings.step_size.y == Approx(10.5f));
        REQUIRE(settings.step_size.z == Approx(10.5f));
        for (int i = 0; i < settings.pyramid_stop_level; ++i) {
            REQUIRE(settings.regularization_weights[i] == Approx(float(i)));
        }
        REQUIRE(settings.constraints_weight == Approx(1234.1234f));

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
        REQUIRE(settings.image_slots[0].cost_functions[0].weight == 
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
