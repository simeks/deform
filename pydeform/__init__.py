import json
import SimpleITK as sitk
import _pydeform

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

    constraint_valuew: sitk.Image
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
    moving_spacing = moving_images[0].GetSpacing()

    fixed_images = [sitk.GetArrayFromImage(img) for img in fixed_images]
    moving_images = [sitk.GetArrayFromImage(img) for img in moving_images]

    if initial_displacement:
        initial_displacement = sitk.GetArrayFromImage(initial_displacement)
    if constraint_mask:
        constraint_mask = sitk.GetArrayFromImage(constraint_mask)
    if constraint_values:
        constraint_values = sitk.GetArrayFromImage(constraint_values)

    settings_str = json.dumps(settings) if settings else ''

    displacement = _pydeform.register(fixed_images,
                                      moving_images,
                                      fixed_spacing,
                                      moving_spacing,
                                      initial_displacement,
                                      constraint_mask,
                                      constraint_values,
                                      settings_str,
                                      num_threads)

    displacement = sitk.GetImageFromArray(displacement)
    displacement.SetOrigin(fixed_origin)
    displacement.SetSpacing(fixed_spacing)

    return displacement

