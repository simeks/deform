import json
import multiprocessing as mp
import SimpleITK as sitk
import _pydeform


def _registration_worker(q, kwargs):
    """ To be ran in a subprocess. """
    try:
        result = _pydeform.register(**kwargs)
    except BaseException as e:
        result = e
    q.put(result)


def register(
        fixed_images,
        moving_images,
        initial_displacement = None,
        constraint_mask = None,
        constraint_values = None,
        settings = None,
        num_threads = 0,
        ):
    """ Perform deformable registration.

    Parameters
    ----------
    fixed_images: Union[sitk.Image, List[sitk.Image]]
        Fixed image, or list of fixed images.

    moving_images: Union[sitk.Image, List[sitk.Image]]
        Moving image, or list of moving images.

    initial_displacement: sitk.Image
        Initial guess of the displacement field.

    constraint_mask: sitk.Image
        Boolean mask for the constraints on the displacement.
        Requires to provide `constraint_values`.

    constraint_values: sitk.Image
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
    sitk.Image
        Vector field containing the displacement that
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

    fixed_origin = fixed_images[0].GetOrigin()
    fixed_spacing = fixed_images[0].GetSpacing()
    moving_origin = moving_images[0].GetOrigin()
    moving_spacing = moving_images[0].GetSpacing()

    fixed_images = [sitk.GetArrayViewFromImage(img) for img in fixed_images]
    moving_images = [sitk.GetArrayViewFromImage(img) for img in moving_images]

    if initial_displacement:
        initial_displacement = sitk.GetArrayViewFromImage(initial_displacement)
    if constraint_mask:
        constraint_mask = sitk.GetArrayViewFromImage(constraint_mask)
    if constraint_values:
        constraint_values = sitk.GetArrayViewFromImage(constraint_values)

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

    # Convert the result to SimpleITK
    displacement = sitk.GetImageFromArray(result)
    displacement.SetOrigin(fixed_origin)
    displacement.SetSpacing(fixed_spacing)

    return displacement

