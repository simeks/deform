import SimpleITK as sitk
from . import numpy_api


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

    # Get numpy view of the input
    fixed_images = [sitk.GetArrayViewFromImage(img) for img in fixed_images]
    moving_images = [sitk.GetArrayViewFromImage(img) for img in moving_images]

    if initial_displacement:
        initial_displacement = sitk.GetArrayViewFromImage(initial_displacement)
    if constraint_mask:
        constraint_mask = sitk.GetArrayViewFromImage(constraint_mask)
    if constraint_values:
        constraint_values = sitk.GetArrayViewFromImage(constraint_values)

    # Perform registration through the numpy API
    displacement = numpy_api.register(fixed_images=fixed_images,
                                      moving_images=moving_images,
                                      fixed_origin=fixed_origin,
                                      moving_origin=moving_origin,
                                      fixed_spacing=fixed_spacing,
                                      moving_spacing=moving_spacing,
                                      initial_displacement=initial_displacement,
                                      constraint_mask=constraint_mask,
                                      constraint_values=constraint_values,
                                      settings=settings,
                                      num_threads=num_threads)

    # Convert the result to SimpleITK
    displacement = sitk.GetImageFromArray(displacement)
    displacement.SetOrigin(fixed_origin)
    displacement.SetSpacing(fixed_spacing)

    return displacement

