# deform

## Prerequisites
* CMake : https://cmake.org/

Optional
* ISPC : https://ispc.github.io/

## Download

Retrieve the repository and associated dependencies by running

```
$ git clone https://github.com/simeks/deform.git
$ cd deform
$ git submodule update --init --recursive
```

## Build

Use CMake (>=3.8) to generate build options of your own choosing.

If CMake cannot find the ISPC executable on your installation, it is possible
to hint the installation directory with `-DISPC_DIR_HINTS`, or to specify the
full path to the executable with `-DISPC_EXECUTABLE`.

### Build options

The build can be configured with the following CMake boolean options:

+ `DF_BUILD_TESTS`: Build unit tests (default: `OFF`)
+ `DF_BUILD_DOCS`: Build Sphinx docs (default: `OFF`)
+ `DF_BUILD_EXECUTABLE`: Build registration executable (default: `ON`)
+ `DF_BUILD_UTILS`: Build utils executable (default: `ON`)
+ `DF_BUILD_PYTHON_WRAPPER`: Build Python wrapper (default: `OFF`)
+ `DF_USE_CUDA`: Enable CUDA support (default: `OFF`)
+ `DF_USE_ISPC`: Enable ISPC support (default: `OFF`)
+ `DF_WARNINGS_ARE_ERRORS`: Warnings are treated as errors (default: `OFF`)
+ `DF_BUILD_WITH_DEBUG_INFO`: Include debug info in release builds (default: `OFF`)
+ `DF_ENABLE_FAST_MATH`: Enable fast math (default: `OFF`)
+ `DF_ITK_BRIDGE`: Add support to interoperate with ITK (default: `OFF`)
+ `DF_STACK_TRACE`: Print a stack trace on errors (default: `OFF`)
+ `DF_ENABLE_MICROPROFILE`: Enable `microprofile` profiler (default: `OFF`)
+ `DF_ENABLE_NVTOOLSEXT`: Enable `nvtoolsext` profiler (default: `OFF`)

## Build and install Python wrapper
```
# python setup.py install
```

Flags accepted by `setup.py`:
* `--use-cuda`: build with CUDA support
* `--use-ispc`: build with ISPC support
* `--use-itk`: build with ITK support
* `--debug`: build with debug symbols

Additional flags starting with `-D` are also recognised and forwarded to CMake.

## Run
To perform a registration
`deform registration -p <param file> -f0 <fixed_0> ... -f<i> <fixed_i> -m0 <moving_0> ... -m<i> <moving_i>`

| Argument                    |                                             |
| --------------------------- | ------------------------------------------- |
| `-f<i> <file>`              | Filename of the i:th fixed image.†          |
| `-m<i> <file>`              | Filename of the i:th moving image.†         |
| `-fm <file>`                | Filename of the fixed mask.‡                |
| `-mm <file>`                | Filename of the moving mask.‡               |
| `-fp <file>`                | Filename for the fixed landmarks.           |
| `-mp <file>`                | Filename for the moving landmarks.          |
| `-d0 <file>`                | Filename for initial deformation field.     |
| `-constraint_mask <file>`   | Filename for constraint mask.               |
| `-constraint_values <file>` | Filename for constraint values.             |
| `-p <file>`                 | Filename of the parameter file.             |
| `-o <file>`                 | Filename of the resulting deformation field |
| `--gpu`                     | Enables GPU assisted registration.          |

† Requires a matching number of fixed and moving images.

‡ Fuzzy masks in floating point format, whose values denote the confidence on
  the image intensity at each point.

### Parameter file example

```yaml
pyramid_levels: 6
pyramid_stop_level: 0
constraints_weight: 1000.0
landmarks_weight: 1.0
landmarks_decay: 2.0
landmarks_stop_level: 0
solver: gc
block_size: [12, 12, 12]
block_energy_epsilon: 1e-7
max_iteration_count: -1
step_size: 0.5
regularizer: diffusion
regularization_weight: 0.1
regularization_scale: 1.0
regularization_exponent: 2.0

levels:
  0:
    regularization_weight: 0.1
  1:
    regularization_weight: 0.2
    step_size: 0.01

image_slots:

  # water
  - resampler: gaussian
    normalize: true
    cost_function:
      - function: ssd
        weight: 0.3
      - function: ncc
        weight: 0.4
        radius: 2
        window: cube
      - function: mi
        weight: 0.6
        sigma: 4.5
        bins: 256
        update_interval: 1
        interpolator: nearest
      - function: gradient_ssd
        weight: 0.7
        sigma: 1.0

  # sfcm
  - resampler: gaussian
    normalize: true
    cost_function: ssd
```

First two parameters, `pyramid_levels` and `pyramid_stop_level`, defines the
size of the pyramid and at which level to stop the registration. Each level
halves the resolution of the input volumes. Setting `pyramid_stop_level` to > 0
specifies that the registration should not be run on the original resolution
(level 0).

`constraints_weight` sets the weight that is applied for constrained voxels. A
really high value means hard constraints while a lower value may allow
constraints to move a certain amount. Cost for constrained voxels are applied
as constraint_weight * squared_distance, where squared_distance is the distance
from the constraint target. See cost function for more info.

`landmarks_weight` sets the weight for the landmark cost term when performing
landmark-based registration.  In order to perform landmark-based registration,
a set of fixed and moving landmarks must be supplied.  The implementation of
the landmark-based unary energy term is inspired to [[3]](#3), but the cost in
each term of the sum is also proportional to the distance between the current
displacement and the landmark displacement. It is possible to limit the usage
of the landmarks up to a certain height of the resolution pyramid by assigning
to `landmarks_stop_level` a value greater than zero. `landmarks_decay` controls
the exponential decay of the landmarks effect with respect to distance in image
space: higher values correspond to faster decay.

`solver` sets the solver for the graph energy minimisation problem. Can be
`graph_cut`, `qpbo`, or `elc`. `graph_cut` operates on cliques up to order 3
and generates a complete labelling, but requires sub-modular terms in order to
run in polynomial time. `qpbo` operates only on binary cliques and can optimise
non-sub-modular terms, in which case it will generate a partial labelling that
is guaranteed to be part of an optimal solution. `elc` can operate on higher
order cliques, reducing the energy to a quadratic form [[1]](#1) and solving
the resulting problem with `qpbo`.

`reduction_mode` sets the algorithm used to reduce higher order terms in the
energy to a quadratic form. Only relevant when using `elc` solver. Available
modes are `hocr`, `elc_hocr`, and `approximate`. For details please refer to
[[1]](#1).

`block_size` size of the block (in voxels) for the block-wise solver. A block
size of (0,0,0) will result in a single block for the whole volume.

`block_energy_epsilon`, minimum percentage decrease of the block energy
required to accept a solution. Higher epsilon will result in lower run time but
also lower quality.

`max_iteration_count`, maximum number of iterations run on each registration
level. Setting this to -1 (default) allows an unlimited number of iterations.

`step_size`, this is the step size in `mm` that the solver will use. Can be a
single `float` value, in that case the same step size will be used in all
directions, or a sequence `[sx, sy, sz]` of three `float` specifying the size
for each direction.

`regularizer` allows to select which regularisation function to use. Available
functions are `diffusion` (minimise first derivative) and `bending` (minimise
second non-mixed derivatives).

`regularization_weight`, `regularization_scale`, and `regularization_exponent`
control the importance of the regularization term. The cost function is
specified as `cost = D + a*((b*R)^c)`, where `D = Σw_i*C_i` is the data term
given by the cost functions `C_i` with weights `w_i`, `R` is the regularization
term, `a` is the regularization weight, `b` the regularization scale, and `c`
the regularization exponent.

`levels`, specifies parameters on a per-level basis. The key indicates which
level the parameters apply to, where 0 is the bottom of the resolution pyramid
(last level). The level identifier can not exceed `pyramid_levels`. Parameters
available on a per-level basis are: `constraints_weight`, `landmarks_weight`,
`block_size`, `block_energy_epsilon`, `max_iteration_count`, `step_size`, and
`regularization_weight`.

`image_slots`, specifies how to use the input images. `resampler` only supports
`gaussian` for now, `normalize` specifies whether the volumes should be
normalized before the registration, and `cost_function` allows to provide one
or more cost functions to use. Its value can be the name of a single function
(`ssd` for squared distance, `ncc` for normalized cross correlation, `mi` for
mutual information, `gradient_ssd` for squared distance of the gradients), in
which case its weight is assumed to be `1.0`, otherwise one or multiple
weighted components can be specified by listing each function and its weight.
Each function can accept a set of parameters.

The parameters available for each function are:
+ `ssd`: no parameters available
+ `ncc`:
  + `window` (`string`): shape of the correlation window, either `sphere` or
      `cube` (default: `spere`). Note that `cube` is available only if the
      program is built with ISPC support. For a given number of samples, the
      sphere has a better spatial distribution of the samples, yielding a
      slightly superior quality. When running on the CPU, for the same number
      of samples (e.g., roughly, a sphere of radius `2` and a cube of radius
      `1`) the cube can be significantly faster to compute.
  + `radius` (`int`): radius of the cross-correlation kernel (default: `2`).
      For `window=sphere`, given a point where NCC is evaluated, samples are
      taken in all the voxels such that the Euclidean distance of each sample
      from the point is lesser or equal to `radius`.  For `window=cube`,
      samples are taken on all voxels within a cube centred on the point and
      with side `2×radius + 1`.
+ `mi`:
  + `bins` (`int`): number of histogram bins used in the approximation of
      probability densities (default: `255`)
  + `sigma` (`float`): standard deviation of the Gaussian kernel used to
      approximate probability densities (default: `4.5`)
  + `update_interval` (`int`): interval (in iterations) between updates of the
      entropy estimates (default: `1`). If `0`, updates are turned off.
  + `interpolator` (`'linear'` or `'nearest'`): interpolator used in the update
      the entropy estimates (default: `'nearest'`)
+ `gradient_ssd`:
  + `sigma` (`float`): Gaussian smoothing applied to the images before
      computing the Sobel operator (default: `0.0`)

### GPU

GPU assisted registration is supported on newer CUDA supported hardware. First
step to enable GPU registration is to compile with the `DF_USE_CUDA=1` flag,
this is set when generating the project with CMake. When both these
prerequisites are met, you simply add the `--gpu` flag to the command-line.

As for now the GPU implementation is considered a pre-release and not all cost
functions and features from the original registration implementation are
supported. Currently the only two supported cost functions are `ssd` and `ncc`.

### Logging

The file name for the log file can be specified through the environment
variable `DF_LOG_FILE`. The minimum level for log messages to be reported can
be set through the environment variable `DF_LOG_LEVEL`, and the possible values
are `Verbose`, `Info`, `Warning`, `Error`, and `Fatal`.

### Masks

It is possible to optionally specify fuzzy masks for the fixed and moving image
space. The two masks can be set independently, and it is possible to use no
mask, only one of the two (either fixed or moving) or both. The masks should be
given in floating point format, and they denote the level of confidence on
image intensity at each voxel. If the mask value `m(x, y, z)` at a certain
location `(x, y, z)` is lesser than or equal to zero, then samples taken at
that location will not contribute to the matching cost. If `m(x, y, z)` is
greater than zero, then the sample will contribute and its cost given at that
location by the image metric will by multiplied by `m(x, y, z)`.

The fixed mask allows to denote a ROI in reference space, formed by all voxels
with strictly positive mask values; for all samples outside such ROI the cost
function will not be computed at all, having the side effect of making the
registration process faster. If a sample belongs to a valid region, then its
mapping through the displacement will be computed and, if a mask for the moving
image is specified, the sample will contribute only if it falls within a valid
ROI in the moving image space, otherwise it will be discarded. The
regularisation term is not weighted by the masks, and it will be always
computed over all the volume, regardless of the mask values.

The moving mask should be used carefully because it can affect the quality of
the result, since there is no penalty for mapping from valid samples in
reference space to regions outside of the moving image mask.

## References

+ <a id="1"></a>[1] Hiroshi Ishikawa: *Higher-order clique reduction without
  auxiliary variables.* Proceedings of the IEEE Conference on Computer Vision
  and Pattern Recognition, 1362-1369. 2014.

+ <a id="2"></a>[2] Junhwan Kim, Vladimir Kolmogorov, Ramin Zabih:
  *Visual correspondence using energy minimization and mutual information.*
  Proceedings of the Ninth IEEE International Conference on Computer Vision,
  1033-1040, 2003.

+ <a id="3"></a>[3] Herve Lombaert, Yiyong Sun, Farida Cheriet:
  *Landmark-based non-rigid registration via graph cuts*,
  International Conference Image Analysis and Recognition, 166–175, 2007
