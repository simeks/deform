import numpy as np
import SimpleITK as sitk
import pydeform
import stk
from . import interruptible

from pydeform import (
    __version__,
    version,
    has_gpu,
    LogLevel
)

def _convert_image(sitk_image):
    """ Converts a SimpleITK.Image to a pydeform.Volume 
    
    Return:
        pydeform.Volume if sitk_image is valid, otherwise None
    """

    if sitk_image is None:
        return None

    return pydeform.Volume(
        sitk.GetArrayViewFromImage(sitk_image),
        sitk_image.GetOrigin(),
        sitk_image.GetSpacing(),
        np.array(sitk_image.GetDirection()).reshape((3,3))
    )

def _convert_transform(transform):
    """ Converts a SimpleITK.AffineTransform to a pydeform.AffineTransform """
    
    translation = np.array(transform.GetTranslation())
    matrix = np.array(transform.GetMatrix()).reshape((3,3))

    # We need to include the fixed parameter (or center) in the offset
    center = np.array(transform.GetCenter())
    offset = translation + center - matrix.dot(center)
    return pydeform.AffineTransform(matrix, offset)


def register(
        fixed_images,
        moving_images,
        *,
        fixed_mask=None,
        moving_mask=None,
        fixed_landmarks=None,
        moving_landmarks=None,
        initial_displacement=None,
        affine_transform=None,
        constraint_mask=None,
        constraint_values=None,
        regularization_map=None,
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

    affine_transform: AffineTransform
        Optional initial affine transformation

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

    # Get numpy view of the input
    fixed_images = [_convert_image(img) for img in fixed_images]
    moving_images = [_convert_image(img) for img in moving_images]

    if None in fixed_images or None in moving_images:
        raise RuntimeError('Cannot pass None as fixed or moving image')


    # A bit of magic since we can't pass None for arguments expecting stk.Volume
    kwargs = {}
    if initial_displacement:
        kwargs['initial_displacement'] = _convert_image(initial_displacement)
    if affine_transform:
        if (not isinstance(affine_transform, sitk.AffineTransform) or 
            affine_transform.GetDimension() != 3):
            raise ValueError(
                'Expected affine transform to be a 3D SimpleITK.AffineTransform'
            )
        kwargs['affine_transform'] = _convert_transform(affine_transform)
    if constraint_mask:
        kwargs['constraint_mask'] = _convert_image(constraint_mask)
    if constraint_values:
        kwargs['constraint_values'] = _convert_image(constraint_values)
    if regularization_map:
        kwargs['regularization_map'] = _convert_image(regularization_map)
    if fixed_mask:
        kwargs['fixed_mask'] = _convert_image(fixed_mask)
    if moving_mask:
        kwargs['moving_mask'] = _convert_image(moving_mask)

    register = interruptible.register if subprocess else pydeform.register

    # Perform registration through the numpy API
    displacement = register(fixed_images=fixed_images,
                            moving_images=moving_images,
                            settings=settings,
                            log=log,
                            log_level=log_level,
                            silent=silent,
                            num_threads=num_threads,
                            use_gpu=use_gpu,
                            **kwargs
                            )

    # Convert the result to SimpleITK
    out = sitk.GetImageFromArray(np.array(displacement, copy=False), isVector=True)
    out.SetOrigin(displacement.origin)
    out.SetSpacing(displacement.spacing)
    out.SetDirection(displacement.direction.astype(np.float64).flatten())

    return out

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
    result = pydeform.jacobian(_convert_image(image))
    result = sitk.GetImageFromArray(np.array(result, copy=False))
    result.CopyInformation(image)
    return result

def regularize(
    displacement,
    precision = 0.5,
    pyramid_levels = 6,
    constraint_mask = None,
    constraint_values = None):
    """Regularize a given displacement field.

    Parameters
    ----------
    displacement: SimpleITK.Image
        Displacement field used to resample the image.
    precision: float
        Amount of precision.
    pyramid_levels: int
        Number of levels for the resolution pyramid
    constraint_mask: SimpleITK.Image
        Mask for constraining displacements in a specific area, i.e., restricting
        any changes within the region.
    constraint_values: SimpleITK.Image
        Vector field specifying the displacements within the constrained regions.

    Returns
    -------
    SimpleITK.Image
        Scalar volume image containing the resulting displacement field.
    """

    if displacement is None:
        raise ValueError('Expected a displacement field')

    displacement = _convert_image(displacement)

    kwargs = {}
    if constraint_mask:
        kwargs['constraint_mask'] = _convert_image(constraint_mask)
    if constraint_values:
        kwargs['constraint_values'] = _convert_image(constraint_values)

    displacement = pydeform.regularize(
        displacement,
        precision,
        pyramid_levels,
        **kwargs
    )

    # Convert the result to SimpleITK
    out = sitk.GetImageFromArray(np.array(displacement, copy=False), isVector=True)
    out.SetOrigin(displacement.origin)
    out.SetSpacing(displacement.spacing)
    out.SetDirection(displacement.direction.astype(np.float64).flatten())
    return out

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
    result = stk.divergence(_convert_image(image))
    result = sitk.GetImageFromArray(np.array(result, copy=False))
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
    result = stk.rotor(_convert_image(image))
    result = sitk.GetImageFromArray(np.array(result, copy=False), isVector=True)
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
    result = stk.circulation_density(_convert_image(image))
    result = sitk.GetImageFromArray(np.array(result, copy=False), isVector=True)
    result.CopyInformation(image)
    return result

