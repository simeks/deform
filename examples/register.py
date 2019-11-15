import sys
import pydeform

"""
This is an example script for registering two images with the cross-correlation
metric using the regular API. Optionally, an affine transform can be provided.
"""

# If pydeform is built with the `--use-cuda` flag, this can be set to
# True to enable GPU asssisted registration. Keep in mind that this is
# still experimental
use_gpu = False

# Settings for the registration
settings = {
    'pyramid_levels': 4, # Perform 4 resolution levels of registration
    'pyramid_stop_level': 0, # Perform all levels up to highest resolution

    # Block size for the optimization, this is a trade-off between quality
    #  and execution time. Larger blocks
    'block_size': [16, 16, 16],
    'max_iteration_count': 50, # Maximum number of iterations each level

    'step_size': 0.25, # Step size in mm

    # How much regularization to apply, energy function is defined as
    # E(I_F, I_M, T) = D(I_F, I_M, T) + a*R(I_F, I_M, T)
    # where 'a' is the regularization weight.
    'regularization_weight': 0.15,

    # Additive update rule, i.e. the displacement field is updated according
    # to `u(x) <- u(x) + delta`
    # This can also be set to compositive for compositive updates, i.e.,
    # `u(x) <- u(x + delta) + delta`
    'update_rule': 'additive',

    # Per level parameters, these parameters override the global parameters
    'levels': {
        '0': {'max_iteration_count': 20},
        '1': {'max_iteration_count': 40},
        '2': {},
        '3': {}
    },

    # Image slot parameters, for this example we only have a single image pair,
    #   to enable additional slots simply add more slots to this list.
    # The order of slots here need to match the order of the images passed to the
    #   registration later on (`pydeform.register([fixed0, fixed1], [moving0, moving1])`)
    'image_slots': [
        {
            # How to resample the images when building the resolution pyramid
            'resampler': 'gaussian',
            # Should the images be normalized prior to the registration
            'normalize': True,
            # Which cost function to use, in this case normalized cross correlation
            'cost_function': 'ncc'
        }
    ]
}

def run(fixed_file, moving_file, output, affine_file=None):
    """ Registers the two files and outputs the transformed moving file """
    
    # Read images using the SimpleITK IO functions
    fixed = pydeform.read_volume(fixed_file)
    moving = pydeform.read_volume(moving_file)

    # To use multiple image pairs you simply replace fixed and moving with lists
    # of images. Remember to add an additional image_slot in the settings.
    # fixed = [pydeform.read_volume((fixed_file0), pydeform.read_volume((fixed_file1)]
    # moving = [pydeform.read_volume((moving_file0), pydeform.read_volume((moving_file1)]

    # Optionally, read an affine transform
    kwargs = {}
    if affine_file:
        kwargs['affine_transform'] = pydeform.read_affine_transform(affine_file)

    # Perform the registration
    with open('registration_log.txt', 'w') as f:
        df = pydeform.register(
            fixed,
            moving,
            settings=settings,
            log=f, # Log output
            log_level=pydeform.LogLevel.Verbose, # Log everything
            use_gpu=use_gpu,
            **kwargs
        )

    # Transform with linear interpolation
    transformed = pydeform.transform(moving, df, pydeform.Interpolator.Linear)

    # Write transformed image to file
    pydeform.write_volume(output, transformed)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: {} <fixed file> <moving file> <output> [affine file]'.format(sys.argv[0]))
        sys.exit(1)

    fixed_file = sys.argv[1]
    moving_file = sys.argv[2]
    output = sys.argv[3]

    affine_file = None
    if len(sys.argv) > 4:
        affine_file = sys.argv[4]

    run(fixed_file, moving_file, output, affine_file)

