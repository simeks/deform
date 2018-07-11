#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>

#include <deform_lib/jacobian.h>

#include <deform_lib/registration/settings.h>
#include <deform_lib/registration/registration.h>
#include <deform_lib/registration/registration_engine.h>
#include <deform_lib/registration/transform.h>

namespace py = pybind11;


/*!
 * \brief Get the shape of the array as an std::vector.
 */
static std::vector<ptrdiff_t> get_shape(const py::array& image) {
    std::vector<ptrdiff_t> shape;
    for (py::ssize_t i = 0; i < image.ndim(); ++i) {
        shape.push_back(image.shape()[i]);
    }
    return shape;
}


/*!
 * \brief Get the scalar shape of the array as an std::vector.
 *
 * Given an array representing a vector volume image, return the
 * shape of the volume.
 */
static std::vector<ptrdiff_t> get_scalar_shape(const py::array& image) {
    auto shape = get_shape(image);
    shape.pop_back();
    return shape;
}


/*!
 * \brief Get the stk::Type associated to a numpy image.
 */
stk::Type get_stk_type(const py::array& a) {
    stk::Type base_type = stk::Type_Unknown;

    if (py::isinstance<py::array_t<char>>(a)) {
        base_type = stk::Type_Char;
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
        throw std::invalid_argument("Unsupported type");
    }

    // NOTE: the value of ndim can be ambiguous, e.g.
    // ndim == 3 may be a scalar volume or a vector 2D image...
    return stk::build_type(base_type, a.ndim() == 4 ? 3 : 1);
}


/*!
 * \brief Convert a numpy array to a stk::Volume.
 *
 * The new volume creates a copy of the input
 * image data.
 *
 * @note The numpy array must be C-contiguous, with
 *       [z,y,x] indexing.
 *
 * @param image Array representing a volume image.
 * @param origin Vector of length 3, containing
 *               the (x, y,z) coordinates of the
 *               volume origin.
 * @param spacing Vector of length 3, containing
 *                the (x, y,z) spacing of the
 *                volume.
 * @return A volume representing the same image.
 */
stk::Volume image_to_volume(
        const py::array image,
        const std::vector<double>& origin,
        const std::vector<double>& spacing
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
    stk::Volume volume {size, get_stk_type(image), image.data()};
    volume.set_origin(origin_);
    volume.set_spacing(spacing_);
    return volume;
}


/*!
 * \brief Wrap the registration routine, converting 
 *        the input and the output to the correct object
 *        types.
 *
 * @note All the numpy arrays must be C-contiguous, with
 *       [z,y,x] indexing.
 *
 * @param fixed_images Vector of numpy arrays, one for each
 *                     modality of the fixed image. The
 *                     order must match the moving images.
 * @param moving_images Vector of numpy arrays, one for each
 *                     modality of the moving image. The
 *                     order must match the fixed images.
 * @param fixed_origin Vector of length 3, containing
 *                     the (x, y,z) coordinates of the
 *                     origin for the fixed images.
 * @param fixed_spacing Vector of length 3, containing
 *                      the (x, y,z) spacing of the
 *                      fixed images.
 * @param moving_origin Analogous to `fixed_origin`.
 * @param moving_spacing Analogous to `fixed_spacing`.
 * @param initial_displacement Must be a numpy array. If `None`,
 *                             a zero displacement is used instead.
 * @param constraint_mask Must be a numpy array. If `None`, no
 *                        constraints are used. If not `None`, a
 *                        corresponding `constraint_values` must
 *                        be provided, otherwise an exception
 *                        is thrown.
 * @param constraint_values Must be a numpy array. If `None`, no
 *                          constraints are used. If not `None`, a
 *                          corresponding `constraint_mask` must
 *                          be provided, otherwise an exception
 *                          is thrown.
 * @param settings_str String containing the json dump of the
 *                     settings. If empty, default settings are used.
 * @param num_threads Number of OpenMP threads to be used. If zero,
 *                    the number is determined automatically, usually
 *                    equal to the number of logic processors available
 *                    on the system.
 *
 * @return A numpy array whose shape is `(nz, ny, nx, 3)`, where
 *         `(x, y, z)` is the shape of the fixed image, containing
 *         the displacement field that registers the moving image.
 *
 * @throw ValidationError If the input is not consistent.
 */
py::array registration_wrapper(
        const std::vector<py::array> fixed_images,
        const std::vector<py::array> moving_images,
        const std::vector<double>& fixed_origin,
        const std::vector<double>& moving_origin,
        const std::vector<double>& fixed_spacing,
        const std::vector<double>& moving_spacing,
        const py::object initial_displacement = py::none(),
        const py::object constraint_mask = py::none(),
        const py::object constraint_values = py::none(),
        const std::string settings_str = "",
        const int num_threads = 0
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
    auto shape = get_shape(fixed_images[0]);
    shape.push_back(3l);

    return py::array_t<float>(shape, reinterpret_cast<const float*>(displacement.ptr()));
}


/*!
 * \brief Resample an image with a given
 *        displacement field.
 *
 * @note All the numpy arrays must be C-contiguous, with
 *       [z,y,x] indexing.
 *
 * @param image Tridimensional numpy array containing the
 *              image data.
 * @param displacement Quadridimensional numpy array containing
 *                     the displacement field, in reference space
 *                     coordinates, with indexing [z, y, x, d].
 * @param fixed_origin Vector of length 3, containing the (x, y,z)
 *                     coordinates of the origin for the reference
 *                     space (displacement).
 * @param fixed_spacing Vector of length 3, containing the (x, y,z)
 *                      spacing of the reference space (displacement).
 * @param moving_origin Analogous to `fixed_origin`, for the moving image.
 * @param moving_spacing Analogous to `fixed_spacing`, for the moving image.
 * @param interpolator Interpolator used in the resampling. The values are 
 *                     exposed in Python as an enum class.
 *
 * @return A numpy array representing the resample image, matching in size,
 *         origin, and spacing the input displacement.
 *
 * @throw ValidationError If the input is not consistent.
 */
py::array transform_wrapper(
        const py::array image,
        const py::array displacement,
        const std::vector<double>& fixed_origin,
        const std::vector<double>& moving_origin,
        const std::vector<double>& fixed_spacing,
        const std::vector<double>& moving_spacing,
        const transform::Interp interpolator = transform::Interp_Linear
        )
{
    const stk::Volume image_ = image_to_volume(image, moving_origin, moving_spacing);
    const stk::Volume displacement_ = image_to_volume(displacement, fixed_origin, fixed_spacing);

    stk::Volume result = transform_volume(image_, displacement_, interpolator);

    auto shape = get_scalar_shape(displacement);
    return py::array(image.dtype(), shape, reinterpret_cast<const float*>(result.ptr()));
}


/*!
 * \brief Compute the Jacobian determinant of a vector field. 
 *
 * The Jacobian is computed with first order central differences.
 *
 * @note All the numpy arrays must be C-contiguous, with
 *       [z,y,x] indexing.
 *
 * @param displacement Quadridimensional numpy array containing
 *                     the displacement field, in reference space
 *                     coordinates, with indexing [z, y, x, d].
 * @param origin Vector of length 3, containing the (x, y,z)
 *               coordinates of the origin.
 * @param spacing Vector of length 3, containing the (x, y,z) spacing.
 *
 * @return A numpy array representing the Jacobian determinant (scalar map)
 *         size, origin, and spacing the input displacement.
 *
 * @throw ValidationError If the input is not consistent.
 */
py::array jacobian_wrapper(
        const py::array displacement,
        const std::vector<double>& origin,
        const std::vector<double>& spacing
        )
{
    auto jacobian = calculate_jacobian(image_to_volume(displacement, origin, spacing));
    auto shape = get_scalar_shape(displacement);
    return py::array_t<JAC_TYPE>(shape, reinterpret_cast<const JAC_TYPE*>(jacobian.ptr()));
}


PYBIND11_MODULE(_pydeform, m)
{
    py::enum_<transform::Interp>(m, "Interpolator")
        .value("NearestNeighbour", transform::Interp_NN)
        .value("Linear", transform::Interp_Linear)
        .export_values();

    m.def("register",
          &registration_wrapper,
          "Perform deformable registration",
          py::arg("fixed_images"),
          py::arg("moving_images"),
          py::arg("fixed_origin") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("moving_origin") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("fixed_spacing") = py::make_tuple(1.0, 1.0, 1.0),
          py::arg("moving_spacing") = py::make_tuple(1.0, 1.0, 1.0),
          py::arg("initial_displacement") = py::none(),
          py::arg("constraint_mask") = py::none(),
          py::arg("constraint_values") = py::none(),
          py::arg("settings_str") = "",
          py::arg("num_threads") = 0
          );

    m.def("transform",
          &transform_wrapper,
          "Deform an image by a displacement field",
          py::arg("image"),
          py::arg("displacement"),
          py::arg("fixed_origin") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("moving_origin") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("fixed_spacing") = py::make_tuple(1.0, 1.0, 1.0),
          py::arg("moving_spacing") = py::make_tuple(1.0, 1.0, 1.0),
          py::arg("interpolator") = transform::Interp_Linear
          );

    m.def("jacobian",
          &jacobian_wrapper,
          "Compute the Jacobian determinant map of a displacement field",
          py::arg("displacement"),
          py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("spacing") = py::make_tuple(1.0, 1.0, 1.0)
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

