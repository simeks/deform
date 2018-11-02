import os
import random
import unittest

import numpy as np
import SimpleITK as sitk

from random import uniform
from numpy.random import rand, randint

import pydeform


# Use a known, random seed for each assert when
# testing with random data.
def _set_seed():
    seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    random.seed(seed)
    return seed


def _gauss3(size=(200, 200, 200), mu=(100, 100, 100), sigma=20, gamma=1):
    x = np.linspace(0, size[2], size[2])
    y = np.linspace(0, size[1], size[1])
    z = np.linspace(0, size[0], size[0])
    x, y, z = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    return gamma * np.exp(-((x-mu[2])/sigma)**2 - ((y-mu[1])/sigma)**2 - ((z-mu[0])/sigma)**2)


def _show(image, origin=(0, 0, 0), spacing=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    image = sitk.GetImageFromArray(image)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    sitk.Show(image)


def _jaccard(a, b):
    return np.sum(np.logical_and(a, b)) / np.sum(np.logical_or(a, b))


def _divergence(f, spacing=(1, 1, 1)):
    dfx_dx = np.gradient(f[..., 0], spacing[0], axis=2)
    dfy_dy = np.gradient(f[..., 1], spacing[1], axis=1)
    dfz_dz = np.gradient(f[..., 2], spacing[2], axis=0)
    return dfx_dx + dfy_dy + dfz_dz


def _rotor(f, spacing=(1, 1, 1)):
    dfx_dy = np.gradient(f[..., 0], spacing[1], axis=1)
    dfx_dz = np.gradient(f[..., 0], spacing[2], axis=0)
    dfy_dx = np.gradient(f[..., 1], spacing[0], axis=2)
    dfy_dz = np.gradient(f[..., 1], spacing[2], axis=0)
    dfz_dx = np.gradient(f[..., 2], spacing[0], axis=2)
    dfz_dy = np.gradient(f[..., 2], spacing[1], axis=1)
    rot = [dfz_dy - dfy_dz, dfx_dz - dfz_dx, dfy_dx - dfx_dy]
    return np.stack(rot, axis=3)


def _circulation_density(f, spacing=(1, 1, 1)):
    return np.linalg.norm(_rotor(f, spacing), axis=3)


def _transform(img,
               d,
               fixed_origin=(0, 0, 0),
               fixed_spacing=(1, 1, 1),
               fixed_direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
               moving_origin=(0, 0, 0),
               moving_spacing=(1, 1, 1),
               moving_direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
               interpolator=sitk.sitkLinear
               ):
    img = sitk.GetImageFromArray(img)
    img.SetOrigin(moving_origin)
    img.SetSpacing(moving_spacing)
    img.SetDirection(moving_direction)

    d = sitk.GetImageFromArray(d)
    d.SetOrigin(fixed_origin)
    d.SetSpacing(fixed_spacing)
    d.SetDirection(fixed_direction)

    warp = sitk.WarpImageFilter()
    warp.SetOutputParameteresFromImage(d)
    warp.SetInterpolator(interpolator)
    res = warp.Execute(img, d)

    return sitk.GetArrayFromImage(res)


class Test_Numpy_API(unittest.TestCase):

    def test_register(self):

        with self.assertRaises(RuntimeError):
            pydeform.register(None, None)

        fixed = rand(10, 10, 10)
        with self.assertRaises(RuntimeError):
            pydeform.register(fixed, None)

        moving = rand(10, 10, 10)
        with self.assertRaises(RuntimeError):
            pydeform.register(None, moving)

        fixed = rand(10, 10, 10)
        moving = [fixed, fixed]
        with self.assertRaises(ValueError):
            pydeform.register(fixed, moving)

        moving = rand(10, 10, 10)
        fixed = [moving, moving]
        with self.assertRaises(ValueError):
            pydeform.register(fixed, moving)

        fixed = (_gauss3(size=(40, 50, 60), mu=(20, 25, 30), sigma=8) > 0.3).astype(np.float32)
        moving = (_gauss3(size=(40, 50, 60), mu=(30, 20, 25), sigma=8) > 0.3).astype(np.float32)

        settings = {
            'regularization_weight': 0.05,
        }

        d = pydeform.register(fixed, moving, settings=settings)

        res = _transform(moving, d, interpolator=sitk.sitkNearestNeighbor)

        self.assertGreater(_jaccard(res > 0.1, fixed > 0.1), 0.98)

    def test_transform(self):
        for _ in range(100):
            seed = _set_seed()

            # Generate some random image data
            pad = 5
            fixed_origin = [uniform(-5, 5) for i in range(3)]
            fixed_spacing = [uniform(0.1, 5) for i in range(3)]
            moving_origin = [uniform(-5, 5) for i in range(3)]
            moving_spacing = [uniform(0.1, 5) for i in range(3)]
            shape_no_pad = [randint(50, 80) for i in range(3)]
            img = np.pad(rand(*shape_no_pad), pad, 'constant')
            d = 5 * (2.0 * rand(*shape_no_pad, 3) - 1.0)
            d = np.pad(d, 3 * [(pad, pad)] + [(0, 0)], 'constant')

            # SimpleITK oracle
            res_sitk = _transform(img,
                                  d,
                                  fixed_origin=fixed_origin,
                                  fixed_spacing=fixed_spacing,
                                  moving_origin=moving_origin,
                                  moving_spacing=moving_spacing,
                                  interpolator=sitk.sitkLinear,
                                  )

            # Compute transform
            res = pydeform.transform(img,
                                     d,
                                     fixed_origin=fixed_origin,
                                     fixed_spacing=fixed_spacing,
                                     moving_origin=moving_origin,
                                     moving_spacing=moving_spacing,
                                     interpolator=pydeform.Interpolator.Linear,
                                     )

            np.testing.assert_almost_equal(res, res_sitk, decimal=4,
                                           err_msg='Mismatch between `transform` and sitk, seed %d' % seed)

    def test_jacobian(self):
        for _ in range(100):
            seed = _set_seed()

            # Generate some random image data
            pad = 1
            origin = [uniform(-5, 5) for i in range(3)]
            spacing = [uniform(0.1, 5) for i in range(3)]
            shape_no_pad = [randint(50, 80) for i in range(3)]
            d = 5 * (2.0 * rand(*shape_no_pad, 3) - 1.0)
            d = np.pad(d, 3 * [(pad, pad)] + [(0, 0)], 'constant')

            # SimpleITK oracle
            d_sitk = sitk.GetImageFromArray(d)
            d_sitk.SetOrigin(origin)
            d_sitk.SetSpacing(spacing)
            jacobian_sitk = sitk.DisplacementFieldJacobianDeterminant(d_sitk)
            jacobian_sitk = sitk.GetArrayFromImage(jacobian_sitk)

            # Compute Jacobian
            jacobian = pydeform.jacobian(d, origin, spacing)

            np.testing.assert_almost_equal(jacobian, jacobian_sitk, decimal=2,
                                           err_msg='Mismatch between `jacobian` and sitk, seed %d' % seed)

    def test_divergence(self):
        for _ in range(100):
            seed = _set_seed()

            # Generate some random image data
            pad = 2
            spacing = [uniform(0.1, 5) for i in range(3)]
            shape_no_pad = [randint(50, 80) for i in range(3)]
            d = 5 * (2.0 * rand(*shape_no_pad, 3) - 1.0)
            d = np.pad(d, 3 * [(pad, pad)] + [(0, 0)], 'constant')

            # NumPy oracle
            divergence_numpy = _divergence(d, spacing=spacing)

            # Compute divergence
            divergence = pydeform.divergence(d, spacing=spacing)

            np.testing.assert_almost_equal(divergence, divergence_numpy, decimal=2,
                                           err_msg='Mismatch between `divergence` and numpy, seed %d' % seed)

    def test_rotor(self):
        for _ in range(100):
            seed = _set_seed()

            # Generate some random image data
            pad = 2
            spacing = [uniform(0.1, 5) for i in range(3)]
            shape_no_pad = [randint(50, 80) for i in range(3)]
            d = 5 * (2.0 * rand(*shape_no_pad, 3) - 1.0)
            d = np.pad(d, 3 * [(pad, pad)] + [(0, 0)], 'constant')

            # NumPy oracle
            rotor_numpy = _rotor(d, spacing=spacing)

            # Compute rotor
            rotor = pydeform.rotor(d, spacing=spacing)

            np.testing.assert_almost_equal(rotor, rotor_numpy, decimal=2,
                                           err_msg='Mismatch between `rotor` and numpy, seed %d' % seed)

    def test_circulation_density(self):
        for _ in range(100):
            seed = _set_seed()

            # Generate some random image data
            pad = 2
            spacing = [uniform(0.1, 5) for i in range(3)]
            shape_no_pad = [randint(50, 80) for i in range(3)]
            d = 5 * (2.0 * rand(*shape_no_pad, 3) - 1.0)
            d = np.pad(d, 3 * [(pad, pad)] + [(0, 0)], 'constant')

            # NumPy oracle
            cd_numpy = _circulation_density(d, spacing=spacing)

            # Compute circulation density
            cd = pydeform.circulation_density(d, spacing=spacing)

            np.testing.assert_almost_equal(cd, cd_numpy, decimal=2,
                                           err_msg='Mismatch between `circulation_density` and numpy, '
                                                   'seed %d' % seed)


if __name__ == '__main__':
    unittest.main()

