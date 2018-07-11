import _pydeform

__all__ = ["numpy_api", "sitk_api"]

Interpolator = _pydeform.Interpolator
""" Enum class for the interpolator functions.

Attributes
----------
Linear: int
    Use trilinear interpolation.
NearestNeighbour: int
    Use nearest neighbour interpolation.
"""
