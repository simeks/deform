#include "cost_function.h"
#include "settings.h"

#include <stk/common/log.h>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4127)
#endif

#include <yaml-cpp/yaml.h>

#ifdef _MSC_VER
    #pragma warning(pop)
#endif


#include <fstream>

class UnrecognisedCostFunction : public YAML::RepresentationException {
 public:
  UnrecognisedCostFunction(const YAML::Mark& mark_)
      : RepresentationException(mark_, "invalid cost function") {}
  UnrecognisedCostFunction(const UnrecognisedCostFunction&) = default;
  virtual ~UnrecognisedCostFunction() noexcept;
};

class UnrecognisedResampleMethod : public YAML::RepresentationException {
 public:
  UnrecognisedResampleMethod(const YAML::Mark& mark_)
      : RepresentationException(mark_, "invalid resample method") {}
  UnrecognisedResampleMethod(const UnrecognisedResampleMethod&) = default;
  virtual ~UnrecognisedResampleMethod() noexcept;
};


/*
pyramid_levels: 6
pyramid_stop_level: 0

constraints_weight: 1000

block_size: [12, 12, 12]
block_energy_epsilon: 0.001
step_size: 0.5
regularization_weight: 0.05

image_slots:

  # water
  - id: 0
    resampler: gaussian
    normalize: true
    cost_function: squared_distance

  # fat
  - id: 1
    resampler: gaussian
    normalize: true
    cost_function: squared_distance

  # sfcm
  - id: 2
    resampler: gaussian
    normalize: true
    cost_function: squared_distance

cost_functions:
  - type: ssd
    weight: 0.5
    slot: 0

  - type: ssd
    weight: 0.5
    slot: 1

  - type: ncc
    weight: 0.5
    radius: 4
    slot: 2


*/

namespace {
    enum CostFunction
    {
        CostFunction_None = 0,
        CostFunction_SSD,
        CostFunction_NCC
    };

}


Settings::ImageSlot::ResampleMethod read_resample_method(const YAML::Node& obj)
{
    std::string fn = obj.as<std::string>();
    if (fn == "gaussian") {
        return Settings::ImageSlot::Resample_Gaussian;
    }

    throw UnrecognisedCostFunction(obj.Mark());
}

CostFunction read_cost_function(const YAML::Node& node)
{
    std::string fn = node.as<std::string>();
    if (fn == "none") {
        return CostFunction_None;
    }
    else if (fn == "squared_distance" || fn == "ssd") {
        return CostFunction_SSD;
    }
    else if (fn == "ncc") {
        return CostFunction_NCC;
    }

    throw UnrecognisedCostFunction(node.Mark());
}

bool read_cost_functions(const YAML::Node& cost_functions, UnaryFunction& unary_fn) {
    if (!cost_functions) {
        return false;
    }

    try {
        if (cost_functions && cost_functions.IsSequence()) {
            for (int i = 0; i < cost_functions.size(); ++i) {
                auto function = cost_functions[i];

                CostFunction type = read_cost_function(function["type"]);
                float weight = function["weight"].as<float>(1.0f);
                int slot = function["slot"].as<int>(-1);

                if (slot < 0 && slot >= DF_MAX_IMAGE_PAIR_COUNT) {
                    LOG(Error) << "Invalid 'slot' provided for cost function, expected slot in range [0, " 
                                << DF_MAX_IMAGE_PAIR_COUNT-1 << "]";
                    return false;
                }

                
            }
        }
    }
    catch (UnrecognisedCostFunction& e) {
        LOG(Error) << "Settings: " << e.what();
        return false;
    }

        slot.cost_functions.resize(slot_node.size());
        for(auto it = slot_node.begin(); it != slot_node.end(); ++it) {
            try {
                slot.cost_functions[k].function = str_to_cost_function(it->first.as<std::string>());
                slot.cost_functions[k].weight = it->second.as<float>();
                ++k;
            }
            catch (YAML::TypedBadConversion<std::string>&) {
                LOG(Error) << "Settings: expected a cost function";
                return false;
            }
            catch (YAML::TypedBadConversion<float>&) {
                LOG(Error) << "Settings: expected a weight (float number)";
                return false;
            }
            catch (UnrecognisedCostFunction& e) {
                LOG(Error) << "Settings: Unrecognized cost function '" << e.what() << "'.";
                return false;
            }
        }
        return true;
    }

    // or it can be a scalar, i.e. a single function (with implicit weight 1.0)
    try {
        slot.cost_functions[0].function = str_to_cost_function(slot_node.as<std::string>());
        slot.cost_functions[0].weight = 1.0f;
    }
    catch (YAML::TypedBadConversion<std::string>&) {
        LOG(Error) << "Settings: expected a cost function";
        return false;
    }
    catch (UnrecognisedCostFunction& e) {
        LOG(Error) << "Settings: Unrecognized cost function '" << e.what() << "'.";
        return false;
    }
    return true;
}

const char* cost_function_to_str(Settings::ImageSlot::CostFunction fn)
{
    switch (fn) {
    case Settings::ImageSlot::CostFunction_SSD:
        return "ssd";
    case Settings::ImageSlot::CostFunction_NCC:
        return "ncc";
    default:
    case Settings::ImageSlot::CostFunction_None:
        return "none";
    }
}
const char* resample_method_to_str(Settings::ImageSlot::ResampleMethod fn)
{
    switch (fn) {
    case Settings::ImageSlot::Resample_Gaussian:
        return "gaussian";
    };
    return "none";
}

void parse_parameters_from_string(const std::string& str, RegistrationInput& input)
{
    YAML::Node root;
    try {
        root = YAML::Load(str);
    }
    catch (YAML::ParserException& e) {
        LOG(Error) << "[YAML] " << e.what();
        return false;
    }
    
    Settings& settings = input.settings;

    try {
        if (root["pyramid_levels"])
            settings.num_pyramid_levels = root["pyramid_levels"].as<int>();
        
        if (root["pyramid_stop_level"])
            settings.pyramid_stop_level = root["pyramid_stop_level"].as<int>();

        if (root["step_size"])
            settings.step_size = root["step_size"].as<float>();
        
        if (root["regularization_weight"])
            settings.regularization_weight = root["regularization_weight"].as<float>();

        if (root["block_energy_epsilon"])
            settings.block_energy_epsilon = root["block_energy_epsilon"].as<double>();

        auto block_size = root["block_size"];
        if (block_size) {
            if (!block_size.IsSequence() || block_size.size() != 3) {
                LOG(Error) << "Settings: 'block_size', expected an array of 3 integers.";
                return false;
            }
            
            settings.block_size = {
                block_size[0].as<int>(),
                block_size[1].as<int>(),
                block_size[2].as<int>()
            };
        }

        if (root["constraints_weight"])
            settings.constraints_weight = root.as<float>("constraints_weight");
        
        auto image_slots = root["image_slots"];
        if (image_slots && image_slots.IsSequence()) {
            for (int i = 0; i < DF_MAX_IMAGE_PAIR_COUNT; ++i) {
                auto slot = image_slots[i];
                if (slot.IsMap()) {
                    int id = slot["id"].as<int>(-1);

                    if (id < 0 && id >= DF_MAX_IMAGE_PAIR_COUNT) {
                        LOG(Error) << "Invalid 'id' provided for image slot, expected id in range [0, " 
                                << DF_MAX_IMAGE_PAIR_COUNT-1 << "]";
                        return false;
                    }

                    if (slot["resampler"])
                        settings.image_slots[id].resample_method = read_resample_method(slot["resampler"]);

                    if (slot["normalize"])
                        settings.image_slots[id].normalize = slot["normalize"].as<bool>();
                }
            }
        }
    }
    catch(YAML::Exception& e) {
        LOG(Error) << "Settings: " << e.what();
        return false;
    }
    
    // Build the cost functions here using the parameter file and the input volumes...


    return true;
}

bool parse_parameter_file(const std::string& parameter_file, RegistrationInput& input)
{
    // Defaults
    settings = Settings();

    std::ifstream f(parameter_file, std::ifstream::in);
    if (!f.is_open()) {
        LOG(Error) << "[Settings] Failed to open file '" << parameter_file << "'";
        return false;
    }

    std::stringstream ss;
    ss << f.rdbuf();

    return initialize_input(ss.str(), input);
}

