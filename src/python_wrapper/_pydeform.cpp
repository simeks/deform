#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>

#include <deform_lib/registration/settings.h>
#include <deform_lib/registration/registration.h>
#include <deform_lib/registration/registration_engine.h>

namespace py = pybind11;


stk::Type get_stk_type(py::array& a) {
    stk::Type base_type = stk::Type_Unknown;

    if (py::isinstance<py::array_t<char>>(a)) {
        base_type =  stk::Type_Char;
    }
    else if (py::isinstance<py::array_t<uint8_t>>(a)) {
        base_type = stk::Type_UChar;
    }
    else if (py::isinstance<py::array_t<short>>(a)) {
        base_type = stk::Type_Short;
    }
    else if (py::isinstance<py::array_t<uint16_t>>(a)) {
        base_type = stk::Type_UShort;
    }
    else if (py::isinstance<py::array_t<int>>(a)) {
        base_type = stk::Type_Int;
    }
    else if (py::isinstance<py::array_t<uint32_t>>(a)) {
        base_type = stk::Type_UInt;
    }
    else if (py::isinstance<py::array_t<float>>(a)) {
        base_type = stk::Type_Float;
    }
    else if (py::isinstance<py::array_t<double>>(a)) {
        base_type = stk::Type_Double;
    }
    else {
        throw std::runtime_error("Unsupported type");
    }

    // NOTE: the value of ndim can be ambiguous, e.g.
    // ndim == 3 may be a scalar volume or a vector 2D image...
    return stk::build_type(base_type, a.ndim() == 4 ? 3 : 1);
}


stk::Volume image_to_volume(
        py::array image,
        std::vector<double>& origin,
        std::vector<double>& spacing
        )
{
    float3 origin_ {
        static_cast<float>(origin[0]),
        static_cast<float>(origin[1]),
        static_cast<float>(origin[2]),
    };
    float3 spacing_ {
        static_cast<float>(spacing[0]),
        static_cast<float>(spacing[1]),
        static_cast<float>(spacing[2]),
    };
    dim3 size {
        static_cast<std::uint32_t>(image.shape(2)),
        static_cast<std::uint32_t>(image.shape(1)),
        static_cast<std::uint32_t>(image.shape(0)),
    };
    stk::Volume volume {size, get_stk_type(image), image.request().ptr};
    volume.set_origin(origin_);
    volume.set_spacing(spacing_);
    return volume;
}


py::array registration_wrapper(
        std::vector<py::array> fixed_images,
        std::vector<py::array> moving_images,
        std::vector<double>& fixed_origin,
        std::vector<double>& moving_origin,
        std::vector<double>& fixed_spacing,
        std::vector<double>& moving_spacing,
        py::object initial_displacement = py::none(),
        py::object constraint_mask = py::none(),
        py::object constraint_values = py::none(),
        std::string settings_str = "",
        int num_threads = 0
        )
{
    // Convert fixed and moving images 
    std::vector<stk::Volume> fixed_volumes, moving_volumes;
    for (size_t i = 0; i < fixed_images.size(); ++i) {
        fixed_volumes.push_back(image_to_volume(fixed_images[i], fixed_origin, fixed_spacing));
        moving_volumes.push_back(image_to_volume(moving_images[i], moving_origin, moving_spacing));
    }

    // Convert optional arguments 
    std::optional<stk::Volume> initial_displacement_;
    if (!initial_displacement.is_none()) {
        initial_displacement_ = image_to_volume(initial_displacement, fixed_origin, fixed_spacing);
    }

    std::optional<stk::Volume> constraint_mask_;
    if (!constraint_mask.is_none()) {
        constraint_mask_ = image_to_volume(constraint_mask, fixed_origin, fixed_spacing);
    }

    std::optional<stk::Volume> constraint_values_;
    if (!constraint_values.is_none()) {
        constraint_values_ = image_to_volume(constraint_values, fixed_origin, fixed_spacing);
    }

    // Parse settings
    Settings settings;
    if ("" != settings_str) {
        parse_registration_settings(settings_str, settings);
    }

    // Perform registration
    stk::Volume displacement = registration(settings,
                                            fixed_volumes,
                                            moving_volumes,
                                            initial_displacement_,
                                            constraint_mask_,
                                            constraint_values_,
                                            num_threads);

    // Build shape
    std::vector<ptrdiff_t> shape;
    for (py::ssize_t i = 0; i < fixed_images[0].ndim(); ++i) {
        shape.push_back(fixed_images[0].shape()[i]);
    }
    shape.push_back(3l);

    return py::array_t<float>(shape, reinterpret_cast<const float*>(displacement.ptr()));
}


PYBIND11_MODULE(_pydeform, m)
{
    m.def("register",
          &registration_wrapper,
          "Perform deformable registration",
          py::arg("fixed_images"),
          py::arg("moving_images"),
          py::arg("fixed_origin"),
          py::arg("moving_origin"),
          py::arg("fixed_spacing"),
          py::arg("moving_spacing"),
          py::arg("initial_displacement") = py::none(),
          py::arg("constraint_mask") = py::none(),
          py::arg("constraint_values") = py::none(),
          py::arg("settings_str") = "",
          py::arg("num_threads") = 0
          );

    // Translate relevant exception types. The exceptions not handled
    // here will be translated autmatically according to pybind11's rules.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        }
        catch (const stk::FatalException &e) {
            // Map stk::FatalException to Python RutimeError
            // +13 to remove the "Fatal error: " from the message
            PyErr_SetString(PyExc_RuntimeError, e.what() + 13);
        }
    });
}

