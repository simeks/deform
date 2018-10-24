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
def set_seed():
    seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    random.seed(seed)
    return seed


def gauss3(size=(200, 200, 200), mu=(100, 100, 100), sigma=20, gamma=1):
    x = np.linspace(0, size[2], size[2])
    y = np.linspace(0, size[1], size[1])
    z = np.linspace(0, size[0], size[0])
    x, y, z = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    return gamma * np.exp(-((x-mu[2])/sigma)**2 - ((y-mu[1])/sigma)**2 - ((z-mu[0])/sigma)**2)


def show(image, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    image = sitk.GetImageFromArray(image)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    sitk.Show(image)


def jaccard(a, b):
    return np.sum(np.logical_and(a, b)) / np.sum(np.logical_or(a, b))


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

        fixed = (gauss3(size=(40, 50, 60), mu=(20, 25, 30), sigma=8) > 0.3).astype(np.float32)
        moving = (gauss3(size=(40, 50, 60), mu=(30, 20, 25), sigma=8) > 0.3).astype(np.float32)

        d = pydeform.register(fixed, moving)

        res = pydeform.transform(moving, d, interpolator=pydeform.Interpolator.NearestNeighbour)

        self.assertGreater(jaccard(res > 0.1, fixed > 0.1), 0.98)

    def test_transform(self):
        for _ in range(100):
            seed = set_seed()

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
            shape = [x + 2*pad for x in shape_no_pad]

            # SimpleITK oracle
            img_sitk = sitk.GetImageFromArray(img)
            img_sitk.SetOrigin(moving_origin)
            img_sitk.SetSpacing(moving_spacing)
            d_sitk = sitk.GetImageFromArray(d)
            d_sitk.SetOrigin(fixed_origin)
            d_sitk.SetSpacing(fixed_spacing)
            d_sitk = sitk.DisplacementFieldTransform(d_sitk)

            res_sitk = sitk.Resample(img_sitk,
                                     list(reversed(shape)),
                                     d_sitk,
                                     sitk.sitkLinear,
                                     fixed_origin,
                                     fixed_spacing,
                                     )
            res_sitk = sitk.GetArrayFromImage(res_sitk)

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
            seed = set_seed()

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


if __name__ == '__main__':
    unittest.main()

