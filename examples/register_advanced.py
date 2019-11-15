import sys
import pydeform
import numpy as np

"""
This is an example script just showcasing some of the more advanced features.

Note:
    This should be seen as an example of the features rather than an example on
    how to produce good registration results.
"""


# Settings for the registration
settings = {
    'pyramid_levels': 4, # Perform 4 resolution levels of registration
    'pyramid_stop_level': 1, # Perform all levels up to highest resolution

    # Block size for the optimization, this is a trade-off between quality
    #  and execution time. Larger blocks
    'block_size': [16, 16, 16],
    'max_iteration_count': 50, # Maximum number of iterations each level

    'step_size': [0.25, 0.25, 0.5], # Step size in mm with larger step on z axis

    # regularization_weight is not used in this example since regularization
    # weights are set by the regularization weight map.
    # 'regularization_weight': 0.15,

    # Further, the regularization term is defined as
    # R(v) = (b*|u(v) - u(w)|^2)^(c/2)
    # where 'b' is the regularization scale and 'c' the regularization exponent
    'regularization_scale': 2.0,
    'regularization_exponent': 4.0,

    # Additive update rule, i.e. the displacement field is updated according
    # to `u(x) <- u(x) + delta`
    # This can also be set to compositive for compositive updates, i.e.,
    # `u(x) <- u(x + delta) + delta`
    'update_rule': 'additive',

    # Per level parameters, these parameters override the global parameters
    'levels': {
        '0': {'max_iteration_count': 20},
        '1': {'max_iteration_count': 40},
        '2': {'step_size': [1, 1, 2]}, # Use a larger step size here
        '3': {'step_size': [2, 2, 4]}
    },

    # Use two image input pairs with NCC and SSD, weight the NCC pair by 0.7
    # and the SSD pair by 0.3
    'image_slots': [
        # First image pair uses ncc
        {
            'resampler': 'gaussian',
            'normalize': True,
            'cost_function': [{
                'function': 'ncc',
                'weight': 0.7
            }]
        },
        # Second pair using ssd
        {
            'resampler': 'gaussian',
            'normalize': True,
            'cost_function': [{
                'function': 'ssd',
                'weight': 0.3
            }]
        }
    ]
}

def build_mask(volume):
    """ Builds a fuzzy mask for the given volume """

    data = np.array(volume, copy=False)
    mask = (data > 0).astype(np.float32)
    mask = pydeform.Volume(mask)
    mask.copy_meta_from(volume)

    return mask
    

def build_regularization_map(volume, threshold, rw0, rw1):
    """ Builds a regularization map given a volume. Voxels with intensities above 
        the given threshold have their regularization set to rw1, while all the
        other voxels have regularization weight rw0.
    """

    data = np.array(volume, copy=False)
    regmap = np.zeros(data.shape, dtype=np.float32)
    regmap = (rw0*(data < threshold) + rw1*(data >= threshold)).astype(np.float32)

    regmap = pydeform.Volume(regmap)
    regmap.copy_meta_from(volume)

    return regmap

def run(fixed0, fixed1, moving0, moving1, output):
    """ Registers the two files and outputs the transformed moving file """
    
    # Read images using the SimpleITK IO functions
    fixed = [
        pydeform.read_volume(fixed0),
        pydeform.read_volume(fixed1)
    ]
    moving = [
        pydeform.read_volume(moving0),
        pydeform.read_volume(moving1)
    ]

    fixed_mask = build_mask(fixed[0])
    moving_mask = build_mask(moving[0])

    regularization_map = build_regularization_map(fixed[0], 100, 0.15, 0.25)

    # Perform the registration
    with open('registration_log.txt', 'w') as f:
        df = pydeform.register(
            fixed,
            moving,
            settings=settings,
            log=f, # Log output
            log_level=pydeform.LogLevel.Verbose, # Log everything
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            regularization_map=regularization_map
        )

    # Transform with linear interpolation
    transformed = pydeform.transform(moving[0], df, pydeform.Interpolator.Linear)

    # Write transformed image to file
    pydeform.write_volume(output, transformed)

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage: {} <fixed0> <fixed1> <moving0> <moving1> <output>'.format(sys.argv[0]))
        sys.exit(1)

    fixed0 = sys.argv[1]
    fixed1 = sys.argv[2]
    moving0 = sys.argv[3]
    moving1 = sys.argv[4]
    output = sys.argv[5]

    run(fixed0, fixed1, moving0, moving1, output)

