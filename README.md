# deform

## Prerequisites
* CMake : https://cmake.org/

## Build
Use CMake (>=3.8) to generate build options of your own choosing.

## Run
To perform a registration
`deform registration -p <param file> -f0 <fixed_0> ... -f<i> <fixed_i> -m0 <moving_0> ... -m<i> <moving_i>`

| Argument                    |                                             |
| --------------------------- | ------------------------------------------- |
| `-f<i> <file>`              | Filename of the i:th fixed image (i < 8)*.  |                                                        
| `-m<i> <file>`              | Filename of the i:th moving image (i < 8)*. |                                                        
| `-d0 <file>`                | Filename for initial deformation field.     |
| `-constraint_mask <file>`   | Filename for constraint mask.               |
| `-constraint_values <file>` | Filename for constraint values.             |
| `-p <file>`                 | Filename of the parameter file.             |
| `-o <file>`                 | Filename of the resulting deformation field |

* Requires a matching number of fixed and moving images.

### Parameter file example

```
{
    "pyramid_levels": 6,
    "pyramid_stop_level": 0,
    "constraints_weight": 1000.0,
    "block_size": [
        12,
        12,
        12
    ],
    "block_energy_epsilon": 0.0001,
    "step_size": 0.5,
    "regularization_weight": 0.1,
    "image_slots": {
        "0": {
            "name": "water",
            "resampler": "gaussian",
            "normalize": true,
            "cost_function": "ssd"
        },
        "1": {
            "name": "sfcm",
            "resampler": "gaussian",
            "normalize": true,
            "cost_function": "ssd"
        }
    }
}
```

First two parameters, `pyramid_levels` and `pyramid_stop_level`, defines the size of the pyramid and at which level to stop the registration. Each level halves the resolution of the input volumes. Setting `pyramid_stop_level` to > 0 specifies that the registration should not be run on the original resolution (level 0).

`constraints_weight` sets the weight that is applied for constrained voxels. A really high value means hard constraints while a lower value may allow constraints to move a certain amount. Cost for constrained voxels are applied as constraint_weight * squared_distance, where squared_distance is the distance from the constraint target. See cost function for more info.

`block_size` size of the block (in voxels) for the block-wise solver. A block size of (0,0,0) will result in a single block for the whole volume.

`block_energy_epsilon`, epsilon applied when determining if a solution is good enough. Higher epsilon will result in lower run time but also lower quality.

`step_size`, this is the step size in [mm] that the solver will use.

`regularization_weight`, value between 0 and 1 used as weight for the regularization term. Cost function is specified as `cost = (1-a)*D + a*R`, where `D` is the data term, `R` is the regularization term, and `a` is the regularization weight.

`image_slots`, specifies how to use the input images. `name` is simply for cosmetic purposes, `resampler` only supports 'gaussian' for now, `normalize` specifies whether the volumes should be normalized before the registration, and `cost_function` specifies which cost function to use ('ssd' for squared distance and 'ncc' for normalized cross correlation).
