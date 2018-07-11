import json
import multiprocessing as mp
import numpy as np
import _pydeform


def _registration_worker(q, kwargs):
    """ To be ran in a subprocess.

    Parameters
    ----------
    q: multiprocessing.Queue
        Queue to return the result.
    kwargs: dict
        Keyword arguments for the registration.
    """
    try:
        result = _pydeform.register(**kwargs)
    except BaseException as e:
        result = e
    q.put(result)


def register(
        fixed_images,
        moving_images,
        fixed_origin = (0, 0, 0),
        moving_origin = (0, 0, 0),
        fixed_spacing = (1, 1, 1),
        moving_spacing = (1, 1, 1),
        initial_displacement = None,
        constraint_mask = None,
        constraint_values = None,
        settings = None,
        num_threads = 0,
        ):
    """ Perform deformable registration.

    ..note:
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

    num_threads: int
        Number of OpenMP threads to be used. If zero, the
        number is selected automatically.

    Returns
    -------
    np.ndarray
        Vector image containing the displacement that
        warps the moving image(s) toward the fixed image(s).
        The displacement is defined in the reference coordinates
        of the fixed image(s), and each voxel contains the
        displacement that allows to resample the voxel from the
        moving image(s).
    """

    if not isinstance(fixed_images, list):
        fixed_images = [fixed_images]
    if not isinstance(moving_images, list):
        moving_images = [moving_images]

    settings_str = json.dumps(settings) if settings else ''

    args = {
        'fixed_images': fixed_images,
        'moving_images': moving_images,
        'fixed_origin': fixed_origin,
        'moving_origin': moving_origin,
        'fixed_spacing': fixed_spacing,
        'moving_spacing': moving_spacing,
        'initial_displacement': initial_displacement,
        'constraint_mask': constraint_mask,
        'constraint_values': constraint_values,
        'settings_str': settings_str,
        'num_threads': num_threads,
    }

    # Run call in a subprocess, to handle keyboard interrupts
    q = mp.Queue()
    p = mp.Process(target=_registration_worker, args=[q, args], daemon=True)
    p.start()
    try:
        result = q.get()
        if isinstance(result, BaseException):
            raise result
        p.join()
    except BaseException as e:
        p.terminate()
        p.join()
        raise e

    return result


transform = _pydeform.transform
""" Warp an image given a displacement field.

The image is resampled using the given displacement field.
The size of the result equals the size of the displacement.

..note:
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

interpolator: pydeform.Interpolator
    Interpolator used in the resampling process, either
    `pydeform.Interpolator.Linear` or
    `pydeform.Interpolator.NearestNeighbour`.

Returns
-------
np.ndarray
    Deformed image obtained resampling the input image
    with the given displacement field.
"""


jacobian = _pydeform.jacobian
""" Compute the Jacobian determinant of the deformation
    associated to a displacement.

Given a displacement field :math:`d(x)`, compute the
Jacobian determinant of its associated deformation field
:math:`D(x) = x + d(x)`.

..note:
    All the arrays must be C-contiguous.

Parameters
----------
displacement: np.ndarray
    Displacement field used to resample the image.

origin: np.ndarray
    Origin of the displacement field.

spacing: np.ndarray
    Spacing of the displacement field.

Returns
-------
np.ndarray
    Scalar volume image containing the Jacobian of the
    deformation associated to the input displacement.
"""

