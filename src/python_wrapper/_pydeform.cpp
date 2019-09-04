#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <map>
#include <string>

#include <stk/filters/vector_calculus.h>

#include <deform_lib/defer.h>
#include <deform_lib/jacobian.h>
#include <deform_lib/make_unique.h>
#include <deform_lib/version.h>

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
 * @param fixed_direction Vector of length 9, representing
 *                        the cosine direction matrix in
 *                        row-major order.
 * @param moving_origin Analogous to `fixed_origin`.
 * @param moving_spacing Analogous to `fixed_spacing`.
 * @param moving_direction Analogous to `fixed_direction`.
 * @param fixed_mask Fixed mask.
 * @param moving_mask Moving mask.
 * @param fixed_landmarks A `n \times 3` numpy array, with
 *                        one row for each landmark point.
 * @param moving_landmarks A `n \times 3` numpy array, with
 *                        one row for each landmark point.
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
 * @param settings Python dictionary for the settings. If `None`,
 *                 default settings are used.
 * @param log Output for the log, either a `StringIO` or a `str` object.
 * @param silent If `True`, do not write output to screen.
 * @param num_threads Number of OpenMP threads to be used. If zero,
 *                    the number is determined automatically, usually
 *                    equal to the number of logic processors available
 *                    on the system.
 * @param use_gpu If `True`, use GPU acceleration from a CUDA device.
 *                Requires a build with CUDA support.
 *
 * @return A numpy array whose shape is `(nz, ny, nx, 3)`, where
 *         `(x, y, z)` is the shape of the fixed image, containing
 *         the displacement field that registers the moving image.
 *
 * @throw ValidationError If the input is not consistent.
 */
py::array registration_wrapper(
        const py::object& fixed_images,
        const py::object& moving_images,
        const std::vector<double>& fixed_origin,
        const std::vector<double>& moving_origin,
        const std::vector<double>& fixed_spacing,
        const std::vector<double>& moving_spacing,
        const std::vector<double>& fixed_direction,
        const std::vector<double>& moving_direction,
        const py::object& fixed_mask,
        const py::object& moving_mask,
        const py::object& fixed_landmarks,
        const py::object& moving_landmarks,
        const py::object& initial_displacement,
        const py::object& constraint_mask,
        const py::object& constraint_values,
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
    std::vector<py::array> fixed_images_;
    if (py::isinstance<py::array>(fixed_images)) {
        fixed_images_ = {py::cast<py::array>(fixed_images)};
    }
    else {
        fixed_images_ = py::cast<std::vector<py::array>>(fixed_images);
    }

    std::vector<py::array> moving_images_;
    if (py::isinstance<py::array>(moving_images)) {
        moving_images_ = {py::cast<py::array>(moving_images)};
    }
    else {
        moving_images_ = py::cast<std::vector<py::array>>(moving_images);
    }

    // Ensure the number of fixed and moving images matches
    if (fixed_images_.size() != moving_images_.size()) {
        throw ValidationError("The number of fixed and moving images must match.");
    }

    // Convert fixed and moving images
    std::vector<stk::Volume> fixed_volumes, moving_volumes;
    for (size_t i = 0; i < fixed_images_.size(); ++i) {
        fixed_volumes.push_back(image_to_volume(fixed_images_[i],
                                                fixed_origin,
                                                fixed_spacing,
                                                fixed_direction));
        moving_volumes.push_back(image_to_volume(moving_images_[i],
                                                  moving_origin,
                                                  moving_spacing,
                                                  moving_direction));
    }

    // Convert optional arguments. Try to cast to the correct numeric
    // type if possible.

    stk::VolumeFloat fixed_mask_;
    if (!fixed_mask.is_none()) {
        fixed_mask_ = image_to_volume(py::cast<py::array_t<float>>(fixed_mask),
                                      fixed_origin,
                                      fixed_spacing,
                                      fixed_direction);
    }

    stk::VolumeFloat moving_mask_;
    if (!moving_mask.is_none()) {
        moving_mask_ = image_to_volume(py::cast<py::array_t<float>>(moving_mask),
                                       moving_origin,
                                       moving_spacing,
                                       moving_direction);
    }

    std::vector<float3> fixed_landmarks_;
    if (!fixed_landmarks.is_none()) {
        fixed_landmarks_ = convert_landmarks(py::cast<py::array_t<float>>(fixed_landmarks));
    }

    std::vector<float3> moving_landmarks_;
    if (!moving_landmarks.is_none()) {
        moving_landmarks_ = convert_landmarks(py::cast<py::array_t<float>>(moving_landmarks));
    }

    stk::Volume initial_displacement_;
    if (!initial_displacement.is_none()) {
        initial_displacement_ = image_to_volume(py::cast<py::array_t<float>>(initial_displacement),
                                                fixed_origin,
                                                fixed_spacing,
                                                fixed_direction);
    }

    stk::Volume constraint_mask_;
    if (!constraint_mask.is_none()) {
        constraint_mask_ = image_to_volume(py::cast<py::array_t<unsigned char>>(constraint_mask),
                                           fixed_origin,
                                           fixed_spacing,
                                           fixed_direction);
    }

    stk::Volume constraint_values_;
    if (!constraint_values.is_none()) {
        constraint_values_ = image_to_volume(py::cast<py::array_t<float>>(constraint_values),
                                             fixed_origin,
                                             fixed_spacing,
                                             fixed_direction);
    }

    // Parse settings
    Settings settings_;
    if (!settings.is_none()) {
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
    if (settings_.image_slots.size() != fixed_images_.size()) {
        LOG(Warning) << "Different number of images between input and settings!";
        settings_.image_slots.resize(fixed_images_.size());
    }

    // Perform registration
    stk::Volume displacement = registration(settings_,
                                            fixed_volumes,
                                            moving_volumes,
                                            fixed_mask_,
                                            moving_mask_,
                                            fixed_landmarks_,
                                            moving_landmarks_,
                                            initial_displacement_,
                                            constraint_mask_,
                                            constraint_values_,
                                            num_threads
                                            #ifdef DF_USE_CUDA
                                            , use_gpu
                                            #endif
                                            );

    // Build shape
    auto shape = get_vector_shape(fixed_images_[0]);

    // This must be done before the `out_stream` goes out of scope
    stk::log_shutdown();

    return py::array_t<float>(shape, reinterpret_cast<const float*>(displacement.ptr()));
}

std::string registration_docstring =
R"(Perform deformable registration.

.. note::
    All the arrays must be C-contiguous.

Parameters
----------
fixed_images: Union[np.ndarray, List[np.ndarray]]
    Fixed image, or list of fixed images.

moving_images: Union[np.ndarray, List[np.ndarray]]
    Moving image, or list of moving images.

fixed_origin: Tuple[Int]
    Origin of the fixed images.

moving_origin: Tuple[Int]
    Origin of the moving images.

fixed_spacing: Tuple[Int]
    Spacing of the fixed images.

moving_spacing: Tuple[Int]
    Spacing of the moving images.

fixed_direction: Tuple[Int]
    Cosine direction matrix of the fixed images.

moving_direction: Tuple[Int]
    Cosine direction matrix of the moving images.

fixed_mask: np.ndarray
    Fixed mask.

moving_mask: np.ndarray
    Moving mask.

fixed_landmarks: np.ndarray
    Array of shape :math:`n \times 3`, with one row
    for each landmark point.

moving_landmarks: np.ndarray
    Array of shape :math:`n \times 3`, with one row
    for each landmark point.

initial_displacement: np.ndarray
    Initial guess of the displacement field.

constraint_mask: np.ndarray
    Boolean mask for the constraints on the displacement.
    Requires to provide `constraint_values`.

constraint_values: np.ndarray
    Value for the constraints on the displacement.
    Requires to provide `constraint_mask`.

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
np.ndarray
    Vector image containing the displacement that
    warps the moving image(s) toward the fixed image(s).
    The displacement is defined in the reference coordinates
    of the fixed image(s), and each voxel contains the
    displacement that allows to resample the voxel from the
    moving image(s).
)";


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
 * @param fixed_direction Vector of length 9, representing
 *                        the cosine direction matrix in
 *                        row-major order.
 * @param moving_origin Analogous to `fixed_origin`, for the moving image.
 * @param moving_spacing Analogous to `fixed_spacing`, for the moving image.
 * @param moving_direction Analogous to `fixed_direction`.
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
        const std::vector<double>& fixed_direction,
        const std::vector<double>& moving_direction,
        const transform::Interp interpolator
        )
{
    const stk::Volume image_ = image_to_volume(image,
                                               moving_origin,
                                               moving_spacing,
                                               moving_direction);

    const stk::Volume displacement_ = image_to_volume(displacement,
                                                      fixed_origin,
                                                      fixed_spacing,
                                                      fixed_direction);

    stk::Volume result = transform_volume(image_, displacement_, interpolator);

    auto shape = get_scalar_shape(displacement);
    return py::array(image.dtype(), shape, reinterpret_cast<const float*>(result.ptr()));
}

std::string transform_docstring =
R"(Warp an image given a displacement field.

The image is resampled using the given displacement field.
The size of the result equals the size of the displacement.

.. note::
    All the arrays must be C-contiguous.

Parameters
----------
image: np.ndarray
    Volume image to be warped.

displacement: np.ndarray
    Displacement field used to resample the image.

fixed_origin: np.ndarray
    Origin of the displacement field.

moving_origin: np.ndarray
    Origin of the moving image.

fixed_spacing: np.ndarray
    Spacing of the displacement field.

moving_spacing: np.ndarray
    Spacing of the moving image.

fixed_direction: Tuple[Int]
    Cosine direction matrix of the fixed images.

moving_direction: Tuple[Int]
    Cosine direction matrix of the moving images.

interpolator: pydeform.Interpolator
    Interpolator used in the resampling process, either
    `pydeform.Interpolator.Linear` or
    `pydeform.Interpolator.NearestNeighbour`.

Returns
-------
np.ndarray
    Deformed image obtained resampling the input image
    with the given displacement field.
)";


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
 * @param direction Vector of length 9, representing the cosine
 *                  direction matrix in row-major order.
 *
 * @return A numpy array representing the Jacobian determinant (scalar map)
 *         of the input displacement.
 *
 * @throw ValidationError If the input is not consistent.
 */
py::array jacobian_wrapper(
        const py::array displacement,
        const std::vector<double>& origin,
        const std::vector<double>& spacing,
        const std::vector<double>& direction
        )
{
    // Compute Jacobian
    auto jacobian = calculate_jacobian(image_to_volume(displacement, origin, spacing, direction));

    auto shape = get_scalar_shape(displacement);
    return py::array_t<JAC_TYPE>(shape, reinterpret_cast<const JAC_TYPE*>(jacobian.ptr()));
}


std::string jacobian_docstring =
R"(Compute the Jacobian determinant of the deformation associated to a displacement.

Given a displacement field :math:`d(x)`, compute the
Jacobian determinant of its associated deformation field
:math:`D(x) = x + d(x)`.

.. note::
    All the arrays must be C-contiguous.

Parameters
----------
displacement: np.ndarray
    Displacement field used to resample the image.

origin: np.ndarray
    Origin of the displacement field.

spacing: np.ndarray
    Spacing of the displacement field.

direction: Tuple[Int]
    Cosine direction matrix of the displacement field.

Returns
-------
np.ndarray
    Scalar volume image containing the Jacobian of the
    deformation associated to the input displacement.
)";


/*!
 * \brief Compute the divergence of a vector field.
 *
 * The divergence is computed with first order central differences.
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
 * @param direction Vector of length 9, representing the cosine
 *                  direction matrix in row-major order.
 *
 * @return A numpy array representing the divergence of the
 *         input displacement.
 *
 * @throw ValidationError If the input is not consistent.
 */
py::array divergence_wrapper(
        const py::array displacement,
        const std::vector<double>& origin,
        const std::vector<double>& spacing,
        const std::vector<double>& direction
        )
{
    // Compute divergence
    auto divergence = stk::divergence<float3>(image_to_volume(displacement, origin, spacing, direction));

    auto shape = get_scalar_shape(displacement);
    return py::array_t<float>(shape, reinterpret_cast<const float*>(divergence.ptr()));
}


std::string divergence_docstring =
R"(Compute the divergence of a displacement field.

The divergence of a vector field

.. math::
    f(\boldsymbol{x}) =
        (f_1(\boldsymbol{x}),
         f_2(\boldsymbol{x}),
         f_3(\boldsymbol{x}))

with :math:`\boldsymbol{x} = (x_1, x_2, x_3)`
is defined as

.. math::
    \nabla \cdot f (\boldsymbol{x}) =
    \sum_{i=1}^3
        \frac{\partial f_i}{\partial x_i} (\boldsymbol{x})

.. note::
    All the arrays must be C-contiguous.

Parameters
----------
displacement: np.ndarray
    Displacement field used to resample the image.

origin: np.ndarray
    Origin of the displacement field.

spacing: np.ndarray
    Spacing of the displacement field.

direction: Tuple[Int]
    Cosine direction matrix of the displacement field.

Returns
-------
np.ndarray
    Scalar volume image containing the divergence of
    the input displacement.
)";


/*!
 * \brief Compute the rotor of a vector field.
 *
 * The rotor is computed with first order central differences.
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
 * @param direction Vector of length 9, representing the cosine
 *                  direction matrix in row-major order.
 *
 * @return A numpy array representing the rotor of the input
 *         displacement.
 *
 * @throw ValidationError If the input is not consistent.
 */
py::array rotor_wrapper(
        const py::array displacement,
        const std::vector<double>& origin,
        const std::vector<double>& spacing,
        const std::vector<double>& direction
        )
{
    // Compute rotor
    auto rotor = stk::rotor<float3>(image_to_volume(displacement, origin, spacing, direction));

    return py::array_t<float>(get_shape(displacement), reinterpret_cast<const float*>(rotor.ptr()));
}


std::string rotor_docstring =
R"(Compute the rotor of a displacement field.

The rotor of a 3D 3-vector field

.. math::
    f(\boldsymbol{x}) =
        (f_1(\boldsymbol{x}),
         f_2(\boldsymbol{x}),
         f_3(\boldsymbol{x}))

with :math:`\boldsymbol{x} = (x_1, x_2, x_3)`
is defined as

.. math::
    \nabla \times f(\boldsymbol{x}) =
    \left(
        \frac{\partial f_3}{\partial x_2} -
        \frac{\partial f_2}{\partial x_3},
        \frac{\partial f_1}{\partial x_3} -
        \frac{\partial f_3}{\partial x_1},
        \frac{\partial f_2}{\partial x_1} -
        \frac{\partial f_1}{\partial x_2}
    \right)

.. note::
    All the arrays must be C-contiguous.

Parameters
----------
displacement: np.ndarray
    Displacement field used to resample the image.

origin: np.ndarray
    Origin of the displacement field.

spacing: np.ndarray
    Spacing of the displacement field.

direction: Tuple[Int]
    Cosine direction matrix of the displacement field.

Returns
-------
np.ndarray
    Vector volume image containing the rotor of
    the input displacement.
)";


/*!
 * \brief Compute the circulation density of a vector field.
 *
 * The circulation density is computed with first order central differences.
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
 * @param direction Vector of length 9, representing the cosine
 *                  direction matrix in row-major order.
 *
 * @return A numpy array representing the circulation density of the input
 *         displacement.
 *
 * @throw ValidationError If the input is not consistent.
 */
py::array circulation_density_wrapper(
        const py::array displacement,
        const std::vector<double>& origin,
        const std::vector<double>& spacing,
        const std::vector<double>& direction
        )
{
    // Compute rotor
    auto res = stk::circulation_density<float3>(image_to_volume(displacement, origin, spacing, direction));

    auto shape = get_scalar_shape(displacement);
    return py::array_t<float>(shape, reinterpret_cast<const float*>(res.ptr()));
}


std::string circulation_density_docstring =
R"(Compute the circulation density of a displacement field.

The circulation density for a 3D 3-vector field is defined as the
norm of the rotor.

.. note::
    All the arrays must be C-contiguous.

Parameters
----------
displacement: np.ndarray
    Displacement field used to resample the image.

origin: np.ndarray
    Origin of the displacement field.

spacing: np.ndarray
    Spacing of the displacement field.

direction: Tuple[Int]
    Cosine direction matrix of the displacement field.

Returns
-------
np.ndarray
    Vector volume image containing the circulation
    density of the input displacement.
)";


PYBIND11_MODULE(_pydeform, m)
{
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

    m.def("register",
          &registration_wrapper,
          registration_docstring.c_str(),
          py::arg("fixed_images"),
          py::arg("moving_images"),
          py::arg("fixed_origin") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("moving_origin") = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("fixed_spacing") = py::make_tuple(1.0, 1.0, 1.0),
          py::arg("moving_spacing") = py::make_tuple(1.0, 1.0, 1.0),
          py::arg("fixed_direction") = py::make_tuple(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
          py::arg("moving_direction") = py::make_tuple(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
          py::arg("fixed_mask") = py::none(),
          py::arg("moving_mask") = py::none(),
          py::arg("fixed_landmarks") = py::none(),
          py::arg("moving_landmarks") = py::none(),
          py::arg("initial_displacement") = py::none(),
          py::arg("constraint_mask") = py::none(),
          py::arg("constraint_values") = py::none(),
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
            py::arg("fixed_origin") = py::make_tuple(0.0, 0.0, 0.0),
            py::arg("moving_origin") = py::make_tuple(0.0, 0.0, 0.0),
            py::arg("fixed_spacing") = py::make_tuple(1.0, 1.0, 1.0),
            py::arg("moving_spacing") = py::make_tuple(1.0, 1.0, 1.0),
            py::arg("fixed_direction") = py::make_tuple(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            py::arg("moving_direction") = py::make_tuple(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            py::arg("interpolator") = transform::Interp_Linear
         );

    m.def("jacobian",
            &jacobian_wrapper,
            jacobian_docstring.c_str(),
            py::arg("displacement"),
            py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0),
            py::arg("spacing") = py::make_tuple(1.0, 1.0, 1.0),
            py::arg("direction") = py::make_tuple(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
         );

    m.def("divergence",
            &divergence_wrapper,
            divergence_docstring.c_str(),
            py::arg("displacement"),
            py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0),
            py::arg("spacing") = py::make_tuple(1.0, 1.0, 1.0),
            py::arg("direction") = py::make_tuple(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
         );

    m.def("rotor",
            &rotor_wrapper,
            rotor_docstring.c_str(),
            py::arg("displacement"),
            py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0),
            py::arg("spacing") = py::make_tuple(1.0, 1.0, 1.0),
            py::arg("direction") = py::make_tuple(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
         );

    m.def("circulation_density",
            &circulation_density_wrapper,
            circulation_density_docstring.c_str(),
            py::arg("displacement"),
            py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0),
            py::arg("spacing") = py::make_tuple(1.0, 1.0, 1.0),
            py::arg("direction") = py::make_tuple(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
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

