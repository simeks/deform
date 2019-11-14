import os
import random
import unittest

import numpy as np
import SimpleITK as sitk

from random import uniform
from numpy.random import rand, randint

import pydeform.sitk_api as pydeform
import _stk as stk

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
    arr = gamma * np.exp(-((x-mu[2])/sigma)**2 - ((y-mu[1])/sigma)**2 - ((z-mu[0])/sigma)**2)
    return sitk.GetImageFromArray(arr.astype(np.float32))


def _show(image, origin=(0, 0, 0), spacing=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    image = sitk.GetImageFromArray(image)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    sitk.Show(image)


def _jaccard(a, b):
    a = sitk.GetArrayFromImage(a)
    b = sitk.GetArrayFromImage(b)
    return np.sum(np.logical_and(a, b)) / np.sum(np.logical_or(a, b))

class Test_SitkAPI(unittest.TestCase):

    def test_register(self):

        with self.assertRaises(RuntimeError):
            pydeform.register(None, None)

        fixed = _gauss3((10, 10, 10))
        with self.assertRaises(RuntimeError):
            pydeform.register(fixed, None)

        moving = _gauss3((10, 10, 10))
        with self.assertRaises(RuntimeError):
            pydeform.register(None, moving)

        fixed = _gauss3((10, 10, 10))
        moving = [fixed, fixed]
        with self.assertRaises(ValueError):
            pydeform.register(fixed, moving)

        moving = _gauss3((10, 10, 10))
        fixed = [moving, moving]
        with self.assertRaises(ValueError):
            pydeform.register(fixed, moving)

        fixed = sitk.Cast(_gauss3(size=(40, 50, 60), mu=(20, 25, 30), sigma=8) > 0.3,
                          sitk.sitkFloat32)
        moving = sitk.Cast(_gauss3(size=(40, 50, 60), mu=(30, 20, 25), sigma=8) > 0.3,
                           sitk.sitkFloat32)

        settings = {
            'regularization_weight': 0.05,
        }

        d = pydeform.register(fixed, moving, settings=settings)

        res = pydeform.transform(moving, d, sitk.sitkNearestNeighbor)

        self.assertGreater(_jaccard(res > 0.1, fixed > 0.1), 0.97)

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
            d_sitk = sitk.GetImageFromArray(d, isVector=True)
            d_sitk.SetOrigin(origin)
            d_sitk.SetSpacing(spacing)
            jacobian_sitk = sitk.DisplacementFieldJacobianDeterminant(d_sitk)
            jacobian_sitk = sitk.GetArrayFromImage(jacobian_sitk)

            # Compute Jacobian
            jacobian = pydeform.jacobian(d_sitk)
            jacobian = sitk.GetArrayFromImage(jacobian)

            np.testing.assert_almost_equal(jacobian, jacobian_sitk, decimal=2,
                                           err_msg='Mismatch between `jacobian` and sitk, seed %d' % seed)


    def test_regularize(self):
        df = sitk.GetImageFromArray(rand(10,10,10,3).astype(np.float32), isVector=True)
        full_mask = sitk.GetImageFromArray(np.ones((10,10,10)).astype(np.uint8))

        out = pydeform.regularize(df)
        # Should not be identical
        self.assertFalse(np.array_equal(np.array(out) , np.array(df)))

        # Should fully replicate constraint values
        constraints = sitk.GetImageFromArray(rand(10,10,10,3).astype(np.float32))
        out = pydeform.regularize(df, constraint_mask=full_mask, constraint_values=constraints)
        np.testing.assert_equal(sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(constraints))

    def test_affine(self):
        # Test affine initialization

        affine_transform = sitk.AffineTransform(3)
        affine_transform.SetTranslation((10,10,10))
        affine_transform.SetMatrix((
            2, 0, 0,
            0, 3, 0,
            0, 0, 4
        ))

        # Do a registration pass without actual iterations to see if affine transform is
        # applied to the resulting displacement field
        settings = {
            'max_iteration_count': 0
        }

        fixed = sitk.GetImageFromArray(np.zeros((10,10,10), dtype=np.float32))
        moving = sitk.GetImageFromArray(np.zeros((10,10,10), dtype=np.float32))

        df = pydeform.register(
            fixed,
            moving,
            settings=settings,
            affine_transform=affine_transform
        )

        df = sitk.GetArrayFromImage(df)

        # Ax + b -> A(1, 1, 1) + b -> (2, 3, 4) + (10, 10, 10) -> (12, 13, 14)
        # u(x) = Ax + b - x
        self.assertEqual(df[1,1,1,0], 11)
        self.assertEqual(df[1,1,1,1], 12)
        self.assertEqual(df[1,1,1,2], 13)

if __name__ == '__main__':
    unittest.main()

