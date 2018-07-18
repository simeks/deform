#pragma once

struct RegistrationInput
{
    Settings settings;

    std::vector<stk::Volume> fixed_volumes;
    std::vector<stk::Volume> moving_volumes;

    std::optional<stk::Volume> initial_deformation;
    std::optional<stk::Volume> constraint_mask;
    std::optional<stk::Volume> constraint_values;

    UnaryFunction unary_fn;
    Regularizer binary_fn;
};

void parse_parameter_file(
    const std::string& parameter_str,
    RegistrationInput& input
);
void parse_parameters_from_string(
    const std::string& parameter_file,
    RegistrationInput& input
);
