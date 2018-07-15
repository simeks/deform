#include "catch.hpp"

#include <deform_lib/registration/settings.h>

#include <cstring>
#include <fstream>

namespace {
const char* sample_settings = R"(
pyramid_levels: 4
pyramid_stop_level: 2

block_size: [20, 24, 38]
block_energy_epsilon: 0.00000000009
step_size: 10.5
regularization_weight: 0.95

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
const char* broken_sample_settings = R"(
pyramid_levels: 4
pyramid_stop_level: 2

block_size: [20, 24
": 10.5
regularization_weight: 0.95

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

TEST_CASE("settings", "")
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
        REQUIRE(settings.step_size == Approx(10.5f));
        REQUIRE(settings.regularization_weight == Approx(0.95f));
        REQUIRE(settings.constraints_weight == Approx(1234.1234f));

        REQUIRE(settings.image_slots[0].resample_method == 
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[0].normalize == true);
        REQUIRE(settings.image_slots[0].cost_function == 
                Settings::ImageSlot::CostFunction_None);

        REQUIRE(settings.image_slots[1].resample_method == 
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[1].normalize == true);
        REQUIRE(settings.image_slots[1].cost_function == 
                Settings::ImageSlot::CostFunction_SSD);
        
        REQUIRE(settings.image_slots[2].resample_method == 
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[2].normalize == true);
        REQUIRE(settings.image_slots[2].cost_function == 
                Settings::ImageSlot::CostFunction_NCC);
                
        REQUIRE(settings.image_slots[3].resample_method == 
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[3].normalize == false);
        REQUIRE(settings.image_slots[3].cost_function == 
                Settings::ImageSlot::CostFunction_None);

        REQUIRE(settings.image_slots[4].resample_method == 
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[4].normalize == false);
        REQUIRE(settings.image_slots[4].cost_function == 
                Settings::ImageSlot::CostFunction_SSD);
        
        REQUIRE(settings.image_slots[5].resample_method == 
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[5].normalize == false);
        REQUIRE(settings.image_slots[5].cost_function == 
                Settings::ImageSlot::CostFunction_NCC);

        REQUIRE(settings.image_slots[6].resample_method == 
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[6].normalize == false);
        REQUIRE(settings.image_slots[6].cost_function == 
                Settings::ImageSlot::CostFunction_SSD);

        REQUIRE(settings.image_slots[7].resample_method == 
                Settings::ImageSlot::Resample_Gaussian);
        REQUIRE(settings.image_slots[7].normalize == false);
        REQUIRE(settings.image_slots[7].cost_function == 
                Settings::ImageSlot::CostFunction_None);
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
