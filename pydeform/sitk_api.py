import SimpleITK as sitk
import pydeform
from . import interruptible


def _sitk2numpy(image):
    R""" Utility to convert a SimpleITK image to numpy.

    .. note::
        A view of the ITK underlying data is returned.
        The array will become invalid if the input
        ITK object is garbage-collected.

    Parameters
    ----------
    image: SimpleITK.Image
        Input SimpleITK object.

    Returns
    -------
    numpy.ndarray
        Array view of the image data.

    Tuple[float]
        Origin of the image.

    Tuple[float]
        Spacing of the image.

    Tuple[float]
        Cosine direction matrix of the image, a :math:`3 \times 3`
        matrix flattened as a tuple (row-major).

    """
    return (
        sitk.GetArrayViewFromImage(image),
        image.GetOrigin(),
        image.GetSpacing(),
        image.GetDirection()
    )


def register(
        fixed_images,
        moving_images,
        *,
        fixed_mask=None,
        moving_mask=None,
        fixed_landmarks=None,
        moving_landmarks=None,
        initial_displacement=None,
        constraint_mask=None,
        constraint_values=None,
        settings=None,
        log=None,
        log_level=pydeform.LogLevel.Info,
        silent=True,
        num_threads=0,
        subprocess=False,
        use_gpu=False
        ):
    R""" Perform deformable registration.

    Parameters
    ----------
    fixed_images: Union[SimpleITK.Image, List[SimpleITK.Image]]
        Fixed image, or list of fixed images.

    moving_images: Union[SimpleITK.Image, List[SimpleITK.Image]]
        Moving image, or list of moving images.

    fixed_mask: np.ndarray
        Fixed mask.

    moving_mask: np.ndarray
        Moving mask.

    fixed_landmarks: np.ndarray
        Array of shape :math:`n \times 3`, containing
        `n` landmarks with (x, y, z) coordinates in
        image space. Requires to provide `moving_landmarks`.

    moving_landmarks: np.ndarray
        Array of shape :math:`n \times 3`, containing
        `n` landmarks with (x, y, z) coordinates in
        image space. Requires to provide `fixed_landmarks`.

    initial_displacement: SimpleITK.Image
        Initial guess of the displacement field.

    constraint_mask: SimpleITK.Image
        Boolean mask for the constraints on the displacement.
        Requires to provide `constraint_values`.

    constraint_values: SimpleITK.Image
        Value for the constraints on the displacement.
        Requires to provide `constraint_mask`.

    settings: dict
        Python dictionary containing the settings for the
        registration.

    log: Union[StringIO, str]
        Output for the log, either a StringIO or a filename.

    log_level: pydeform.LogLevel
        Minimum level for log messages to be reported.

    silent: bool
        If `True`, do not write output to screen.

    num_threads: int
        Number of OpenMP threads to be used. If zero, the
        number is selected automatically.

    subprocess: bool
        If `True`, run the call in a subprocess and handle
        keyboard interrupts. This has a memory overhead, since
        a new instance of the intepreter is spawned and
        input objects are copied in the subprocess memory.

    use_gpu: bool
        If `True`, use GPU acceleration from a CUDA device.
        Requires a build with CUDA support.

    Returns
    -------
    SimpleITK.Image
        Vector field containing the displacement that
        warps the moving image(s) toward the fixed image(s).
        The displacement is defined in the reference coordinates
        of the fixed image(s), and each voxel contains the
        displacement that allows to resample the voxel from the
        moving image(s).
    """

    if not isinstance(fixed_images, (list, tuple)):
        fixed_images = [fixed_images]
    if not isinstance(moving_images, (list, tuple)):
        moving_images = [moving_images]

    fixed_origin = fixed_images[0].GetOrigin()
    fixed_spacing = fixed_images[0].GetSpacing()
    fixed_direction = fixed_images[0].GetDirection()
    moving_origin = moving_images[0].GetOrigin()
    moving_spacing = moving_images[0].GetSpacing()
    moving_direction = moving_images[0].GetDirection()

    # Get numpy view of the input
    fixed_images = [sitk.GetArrayViewFromImage(img) for img in fixed_images]
    moving_images = [sitk.GetArrayViewFromImage(img) for img in moving_images]

    if initial_displacement:
        initial_displacement = sitk.GetArrayViewFromImage(initial_displacement)
    if constraint_mask:
        constraint_mask = sitk.GetArrayViewFromImage(constraint_mask)
    if constraint_values:
        constraint_values = sitk.GetArrayViewFromImage(constraint_values)
    if fixed_mask:
        fixed_mask = sitk.GetArrayViewFromImage(fixed_mask)
    if moving_mask:
        moving_mask = sitk.GetArrayViewFromImage(moving_mask)

    register = interruptible.register if subprocess else pydeform.register

    # Perform registration through the numpy API
    displacement = register(fixed_images=fixed_images,
                            moving_images=moving_images,
                            fixed_origin=fixed_origin,
                            moving_origin=moving_origin,
                            fixed_spacing=fixed_spacing,
                            moving_spacing=moving_spacing,
                            fixed_direction=fixed_direction,
                            moving_direction=moving_direction,
                            fixed_mask=fixed_mask,
                            moving_mask=moving_mask,
                            fixed_landmarks=fixed_landmarks,
                            moving_landmarks=moving_landmarks,
                            initial_displacement=initial_displacement,
                            constraint_mask=constraint_mask,
                            constraint_values=constraint_values,
                            settings=settings,
                            log=log,
                            log_level=log_level,
                            silent=silent,
                            num_threads=num_threads,
                            use_gpu=use_gpu,
                            )

    # Convert the result to SimpleITK
    displacement = sitk.GetImageFromArray(displacement)
    displacement.SetOrigin(fixed_origin)
    displacement.SetSpacing(fixed_spacing)
    displacement.SetDirection(fixed_direction)

    return displacement

def transform(image, df, interp=sitk.sitkLinear):
    R""" Resample an image with a given displacement field.
    
    Parameters
    ----------
    image: SimpleITK.Image
        Image to resample.
        
    df:
        Displacementfield, in reference space coordinates, to apply
    
    interp: pydeform.Interpolator
        Interpolator used in the resampling. E.g. SimpleITK.sitkLinear

    Returns
    -------
    SimpleITK.Image
        Resampled image
    """
    t = sitk.DisplacementFieldTransform(sitk.Cast(df, sitk.sitkVectorFloat64))
    return sitk.Resample(image, df, t, interp)
    

def jacobian(image):
    R""" Compute the Jacobian determinant of a 3D 3-vector field.

    The Jacobian determinant of a vector field

    .. math::
        f(\boldsymbol{x}) =
            (f_1(\boldsymbol{x}),
             f_2(\boldsymbol{x}),
             f_3(\boldsymbol{x}))

    with :math:`\boldsymbol{x} = (x_1, x_2, x_3)`
    is defined as

    .. math::
        J[f] (\boldsymbol{x}) =
        \det \left(
            \frac{\partial f_i}{\partial x_j} (\boldsymbol{x})
        \right)_{ij}

    Parameters
    ----------
    image: SimpleITK.Image
        Input image, should be a 3D 3-vector field.

    Returns
    -------
    SimpleITK.Image
        Scalar image representing the Jacobian determinant.
    """
    result = pydeform.jacobian(*_sitk2numpy(image))
    result = sitk.GetImageFromArray(result)
    result.CopyInformation(image)
    return result


def divergence(image):
    R""" Compute the divergence of a 3D 3-vector field.

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

    Parameters
    ----------
    image: SimpleITK.Image
        Input image, should be a 3D 3-vector field.

    Returns
    -------
    SimpleITK.Image
        Scalar image representing the divergence.
    """
    result = pydeform.divergence(*_sitk2numpy(image))
    result = sitk.GetImageFromArray(result)
    result.CopyInformation(image)
    return result


def rotor(image):
    R""" Compute the rotor of a vector field.

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

    Parameters
    ----------
    image: SimpleITK.Image
        Input image, should be a 3D 3-vector field.

    Returns
    -------
    SimpleITK.Image
        Vector image representing the rotor.
    """
    result = pydeform.rotor(*_sitk2numpy(image))
    result = sitk.GetImageFromArray(result)
    result.CopyInformation(image)
    return result


def circulation_density(image):
    R""" Compute the circulation density of a vector field.

    The circulation density for a 3D 3-vector field is defined as the
    norm of the rotor.

    Parameters
    ----------
    image: SimpleITK.Image
        Input image, should be a 3D 3-vector field.

    Returns
    -------
    SimpleITK.Image
        Vector image representing the circulation density.
    """
    result = pydeform.circulation_density(*_sitk2numpy(image))
    result = sitk.GetImageFromArray(result)
    result.CopyInformation(image)
    return result

