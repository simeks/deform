#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <map>
#include <string>

#include <deform_lib/defer.h>
#include <deform_lib/jacobian.h>
#include <deform_lib/make_unique.h>
#include <deform_lib/regularize.h>
#include <deform_lib/version.h>

#include <deform_lib/registration/affine_transform.h>
#include <deform_lib/registration/displacement_field.h>
#include <deform_lib/registration/settings.h>
#include <deform_lib/registration/registration.h>
#include <deform_lib/registration/registration_engine.h>
#include <deform_lib/registration/transform.h>

#include <stk/image/volume.h>

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
    assert((shape.size() == 4 && "The image is already a scalar volume"));
    shape.pop_back();
    return shape;
}


/*!
 * \brief Get the vector shape of the array as an std::vector.
 *
 * Given an array representing a scalar volume image, return the
 * shape of the volume of an associated vector image.
 */
static std::vector<ptrdiff_t> get_vector_shape(const py::array& image) {
    auto shape = get_shape(image);
    assert((shape.size() == 3 && "The image is already a vector volume"));
    shape.push_back(3l);
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
    else if (py::isinstance<py::array_t<bool>>(a)) {
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
 * @param direction Vector of length 9, representing
 *                  the cosine direction matrix in
 *                  row-major order.
 * @return A volume representing the same image.
 */
stk::Volume image_to_volume(
        const py::array image,
        const std::vector<double>& origin,
        const std::vector<double>& spacing,
        const std::vector<double>& direction
        )
{
    if (image.flags() & py::array::f_style) {
        throw std::invalid_argument("The arrays must be C-contiguous.");
    }
    
    float3 origin_ {
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
    };
    float3 spacing_ {
        float(spacing[0]),
        float(spacing[1]),
        float(spacing[2]),
    };
    Matrix3x3f direction_ {{
        {float(direction[0]), float(direction[1]), float(direction[2])},
        {float(direction[3]), float(direction[4]), float(direction[5])},
        {float(direction[6]), float(direction[7]), float(direction[8])},
    }};
    dim3 size {
        std::uint32_t(image.shape(2)),
        std::uint32_t(image.shape(1)),
        std::uint32_t(image.shape(0)),
    };
    stk::Volume volume {size, get_stk_type(image), image.data()};
    volume.set_origin(origin_);
    volume.set_spacing(spacing_);
    volume.set_direction(direction_);
    return volume;
}

/*!
 * \brief Validate and convert the landmarks.
 *
 * Convert the landmarks to a C++ vector.
 *
 * @note A copy of the landmarks data is created.
 * @note The numpy array must be C-contiguous.
 *
 * @param landmarks Array `n \times 3` representing
 *                  the landmarks.
 * @return A vector of landmarks.
 */
std::vector<float3> convert_landmarks(const py::array_t<float> landmarks_array)
{
    if (landmarks_array.ndim() != 2 || landmarks_array.shape(1) != 3) {
        throw std::invalid_argument("The landmarks must be a `n × 3` array.");
    }

    std::vector<float3> landmarks;
    for (ptrdiff_t i = 0; i < landmarks_array.shape(0); ++i) {
        landmarks.push_back({
                *landmarks_array.data(i, 0),
                *landmarks_array.data(i, 1),
                *landmarks_array.data(i, 2),
                });
    }

    return landmarks;
}

/*!
 * \brief Add the logger, converting it if necessary.
 *
 * If the input is a string, add a file logger using it as filename. If
 * it is a Python object, try to cast it to StringIO and add a stream
 * logger.
 */
void add_logger(
        const py::object& log,
        const stk::LogLevel level,
        std::unique_ptr<py::detail::pythonbuf>& buffer,
        std::unique_ptr<std::ostream>& out_stream
        )
{
    if (log.is_none()) {
        return;
    }

    try {
        stk::log_add_file(py::cast<std::string>(log).c_str(), level);
    }
    catch (py::cast_error &) {
        try {
            buffer = make_unique<py::detail::pythonbuf>(log);
            out_stream = make_unique<std::ostream>(buffer.get());
            stk::log_add_stream(out_stream.get(), level);
        }
        catch (...) {
            throw std::invalid_argument("Invalid log object!");
        }
    }
}

std::string registration_docstring =
R"(Perform deformable registration.

Parameters
----------
fixed_images: Union[stk.Volume, List[stk.Volume]]
    Fixed image, or list of fixed images.

moving_images: Union[stk.Volume, List[stk.Volume]]
    Moving image, or list of moving images.

fixed_mask: stk.Volume
    Fixed mask.

moving_mask: stk.Volume
    Moving mask.

fixed_landmarks: np.ndarray
    Array of shape :math:`n \times 3`, with one row
    for each landmark point.

moving_landmarks: np.ndarray
    Array of shape :math:`n \times 3`, with one row
    for each landmark point.

initial_displacement: stk.Volume
    Initial guess of the displacement field.

affine_transform: AffineTransform
    Initial affine transformation

constraint_mask: stk.Volume
    Boolean mask for the constraints on the displacement.
    Requires to provide `constraint_values`.

constraint_values: stk.Volume
    Value for the constraints on the displacement.
    Requires to provide `constraint_mask`.

regularization_map: stk.Volume
    Map of voxel-wise regularization weights.
    Should be the same shape as the fixed image.

settings: dict
    Python dictionary containing the settings for the
    registration.

log: Union[StringIO, str]
    Output for the log, either a StringIO or a filename.

silent: bool
    If `True`, do not write output to screen.

num_threads: int
    Number of OpenMP threads to be used. If zero, the
    number is selected automatically.

use_gpu: bool
    If `True`, use GPU acceleration from a CUDA device.
    Requires a build with CUDA support.

Returns
-------
stk.Volume
    Vector image containing the displacement that
    warps the moving image(s) toward the fixed image(s).
    The displacement is defined in the reference coordinates
    of the fixed image(s), and each voxel contains the
    displacement that allows to resample the voxel from the
    moving image(s).
)";

stk::Volume registration_wrapper(
        const py::object& fixed_images,
        const py::object& moving_images,
        const stk::Volume& fixed_mask,
        const stk::Volume& moving_mask,
        const py::object& fixed_landmarks,
        const py::object& moving_landmarks,
        const stk::Volume& initial_displacement,
        const AffineTransform& affine_transform,
        const stk::Volume& constraint_mask,
        const stk::Volume& constraint_values,
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
        const stk::Volume& regularization_map,
#endif
        const py::object& settings,
        const py::object& log,
        const stk::LogLevel log_level,
        const bool silent,
        const int num_threads,
        const bool use_gpu
)
{
    #ifndef DF_USE_CUDA
    if (use_gpu) {
        throw std::invalid_argument("This build of pydeform has no CUDA support!");
    }
    #endif

    // Redirect C++ stdout/stderr to Python's
    py::scoped_ostream_redirect std_out(std::cout, py::module::import("sys").attr("stdout"));
    py::scoped_ostream_redirect std_err(std::cerr, py::module::import("sys").attr("stderr"));

    // Handle logging
    std::unique_ptr<py::detail::pythonbuf> buffer;
    std::unique_ptr<std::ostream> out_stream;
    stk::log_init();
    if (!silent)
        stk::log_add_stream(&std::cerr, log_level);
    add_logger(log, log_level, buffer, out_stream);
    LOG(Info) << deform::version_string();

    // Handle single images passed as objects, without a container
    std::vector<stk::Volume> fixed_volumes;
    if (py::isinstance<stk::Volume>(fixed_images)) {
        fixed_volumes = {py::cast<stk::Volume>(fixed_images)};
    }
    else if (py::isinstance<py::array>(fixed_images)) {
        fixed_volumes = {py::cast<stk::Volume>(fixed_images)};
    }
    else {
        fixed_volumes = py::cast<std::vector<stk::Volume>>(fixed_images);
    }

    std::vector<stk::Volume> moving_volumes;
    if (py::isinstance<stk::Volume>(moving_images)) {
        moving_volumes = {py::cast<stk::Volume>(moving_images)};
    }
    else if (py::isinstance<py::array>(moving_images)) {
        moving_volumes = {py::cast<stk::Volume>(moving_images)};
    }
    else {
        moving_volumes = py::cast<std::vector<stk::Volume>>(moving_images);
    }

    // Ensure the number of fixed and moving images matches
    if (fixed_volumes.size() != moving_volumes.size()) {
        throw ValidationError("The number of fixed and moving images must match.");
    }

    // Convert optional arguments. Try to cast to the correct numeric
    // type if possible.

    std::vector<float3> fixed_landmarks_;
    if (!fixed_landmarks.is_none()) {
        fixed_landmarks_ = convert_landmarks(py::cast<py::array_t<float>>(fixed_landmarks));
    }

    std::vector<float3> moving_landmarks_;
    if (!moving_landmarks.is_none()) {
        moving_landmarks_ = convert_landmarks(py::cast<py::array_t<float>>(moving_landmarks));
    }

    // Parse settings
    Settings settings_;
    if (!settings.is_none()) {
        // Convert the python dict into a YAML string, which then is parseable by settings
        py::object py_yaml_dump = py::module::import("yaml").attr("dump");
        py::object py_settings_str = py_yaml_dump(py::cast<py::dict>(settings));
        std::string settings_str = py::cast<std::string>(py_settings_str);
        parse_registration_settings(settings_str, settings_);

        // Print only contents of parameter file to Info
        LOG(Info) << "Parameters:" << std::endl << settings_str;

        std::stringstream settings_ss;
        print_registration_settings(settings_, settings_ss);

        // Print all settings to Verbose
        LOG(Verbose) << settings_ss.rdbuf();
    }

    // Check number of image slots
    if (settings_.image_slots.size() != fixed_volumes.size()) {
        LOG(Warning) << "Different number of images between input and settings!";
        settings_.image_slots.resize(fixed_volumes.size());
    }

    // Perform registration
    stk::Volume displacement = registration(settings_,
                                            fixed_volumes,
                                            moving_volumes,
                                            fixed_mask,
                                            moving_mask,
                                            fixed_landmarks_,
                                            moving_landmarks_,
                                            initial_displacement,
                                            affine_transform,
                                            constraint_mask,
                                            constraint_values,
                                        #ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
                                            regularization_map,
                                        #endif
                                            num_threads
                                        #ifdef DF_USE_CUDA
                                            , use_gpu
                                        #endif
                                            );

    // This must be done before the `out_stream` goes out of scope
    stk::log_shutdown();

    return displacement;
}


std::string transform_docstring =
R"(Warp an image given a displacement field.

The image is resampled using the given displacement field.
The size of the result equals the size of the displacement.

Parameters
----------
image: stk.Volume
    Volume image to be warped.

displacement: stk.Volume
    Displacement field used to resample the image.

interpolator: pydeform.Interpolator
    Interpolator used in the resampling process, either
    `pydeform.Interpolator.Linear` or
    `pydeform.Interpolator.NearestNeighbour`.

affine_transform: AffineTransform
    Optional affine transformation

Returns
-------
stk.Volume
    Deformed image obtained resampling the input image
    with the given displacement field.
)";

stk::Volume transform_wrapper(
    const stk::Volume& src,
    const stk::Volume& df,
    transform::Interp interp,
    const AffineTransform& affine_transform)
{
    return transform_volume(src, DisplacementField(df, affine_transform), interp);
}

std::string jacobian_docstring =
R"(Compute the Jacobian determinant of the deformation associated to a displacement.

Given a displacement field :math:`d(x)`, compute the
Jacobian determinant of its associated deformation field
:math:`D(x) = x + d(x)`.

Parameters
----------
displacement: stk.Volume
    Displacement field used to resample the image.

Returns
-------
stk.Volume
    Scalar volume image containing the Jacobian of the
    deformation associated to the input displacement.
)";

stk::Volume jacobian_wrapper(const stk::Volume& df)
{
    return calculate_jacobian(df);
}

std::string regularization_docstring =
R"(Regularize a given displacement field.

Parameters
----------
displacement: stk.Volume
    Displacement field used to resample the image.
precision: float
    Amount of precision.
pyramid_levels: int
    Number of levels for the resolution pyramid
constraint_mask: stk.Volume
    Mask for constraining displacements in a specific area, i.e., restricting
    any changes within the region.
constraint_values: stk.Volume
    Vector field specifying the displacements within the constrained regions.

Returns
-------
stk.Volume
    Scalar volume image containing the resulting displacement field.
)";


stk::Volume regularization_wrapper(
    const stk::Volume& displacement,
    float precision,
    int pyramid_levels,
    const stk::Volume& constraint_mask,
    const stk::Volume& constraint_values)
{
    return regularization(
        displacement,
        precision,
        pyramid_levels,
        constraint_mask,
        constraint_values
    );
}


AffineTransform make_affine_transform(
    const py::array_t<double>& matrix,
    const py::array_t<double>& offset
)
{
    AffineTransform t;

    if (matrix.ndim() != 2 ||
        matrix.shape(0) != 3 ||
        matrix.shape(1) != 3) {
        throw py::value_error("Invalid shape of affine matrix, expected (3, 3).");
    }

    Matrix3x3f m;
    for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
        m(i, j) = (float)matrix.at(i, j);
    }}

    if (offset.ndim() != 1 ||
        offset.shape(0) != 3) {
        throw py::value_error("Invalid shape of affine offset, expected (3).");
    }

    float3 o {
        (float)offset.at(0),
        (float)offset.at(1),
        (float)offset.at(2)
    };

    return AffineTransform(m, o);
}

PYBIND11_MODULE(_pydeform, m)
{
    m.attr("__version__") = GIT_VERSION_TAG;
    m.def("version", [](){
        return deform::version_string();
    });
    m.def("has_gpu", [](){
        #ifdef DF_USE_CUDA
            return true;
        #else
            return false;
        #endif
    });

    m.import("_stk");

    py::enum_<transform::Interp>(m, "Interpolator", "Interpolator functions")
        .value("NearestNeighbour", transform::Interp_NN, "Nearest neighbour interpolation")
        .value("NearestNeighbor", transform::Interp_NN, "Non-British spelling for NearestNeighbour")
        .value("Linear", transform::Interp_Linear, "Trilinear interpolation")
        .export_values();

    py::enum_<stk::LogLevel>(m, "LogLevel", "Level for the logger")
        .value("Verbose", stk::LogLevel::Verbose, "Lowest level, report all messages")
        .value("Info", stk::LogLevel::Info, "Report informative messages, warnings and errors")
        .value("Warning", stk::LogLevel::Warning, "Report warnings and errors")
        .value("Error", stk::LogLevel::Error, "Report only errors")
        .value("Fatal", stk::LogLevel::Fatal, "Report only fatal errors")
        .export_values();

    py::class_<AffineTransform>(m, "AffineTransform")
        .def(py::init<>())
        .def(py::init(&make_affine_transform),
            py::arg("matrix"),
            py::arg("offset")
        );

    m.def("register",
          &registration_wrapper,
          registration_docstring.c_str(),
          py::arg("fixed_images"),
          py::arg("moving_images"),
          py::arg("fixed_mask") = stk::Volume(),
          py::arg("moving_mask") = stk::Volume(),
          py::arg("fixed_landmarks") = py::none(),
          py::arg("moving_landmarks") = py::none(),
          py::arg("initial_displacement") = stk::Volume(),
          py::arg("affine_transform") = AffineTransform(),
          py::arg("constraint_mask") = stk::Volume(),
          py::arg("constraint_values") = stk::Volume(),
#ifdef DF_ENABLE_REGULARIZATION_WEIGHT_MAP
          py::arg("regularization_map") = stk::Volume(),
#endif
          py::arg("settings") = py::none(),
          py::arg("log") = py::none(),
          py::arg("log_level") = stk::LogLevel::Info,
          py::arg("silent") = true,
          py::arg("num_threads") = 0,
          py::arg("use_gpu") = false
          );

    m.def("transform",
            &transform_wrapper,
            transform_docstring.c_str(),
            py::arg("image"),
            py::arg("displacement"),
            py::arg("interpolator") = transform::Interp_Linear,
            py::arg("affine_transform") = AffineTransform()
         );

    m.def("jacobian",
            &jacobian_wrapper,
            jacobian_docstring.c_str()
         );

    m.def("regularize",
            &regularization_wrapper,
            regularization_docstring.c_str(),
            py::arg("displacement"),
            py::arg("precision") = 0.5f,
            py::arg("pyramid_levels") = 6,
            py::arg("constraint_mask") = stk::Volume(),
            py::arg("constraint_values") = stk::Volume()
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

