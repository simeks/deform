import numpy as np
import pydeform

""" This is just a few examples on how to use the built-in pydeform.Volume type 
"""

def normalize_volume(vol):
    """ Returns a new normalized copy of `vol` """

    assert isinstance(vol, pydeform.Volume)

    # Create a new array with a copy of the volume data
    data = np.array(vol)

    # Create a new normalized array
    data = data / np.max(data)

    # Create a new volume
    out = pydeform.Volume(data)
    # Copy original meta data (origin, spacing, direction)
    out.copy_meta_from(vol)

    return out


def normalize_volume_in_place(vol):
    """ Normalize the volume in place, without copying the data """

    assert isinstance(vol, pydeform.Volume)

    # Convert the volume object into a numpy array
    # `copy=False` means that the array object only holds a reference to the 
    # data within the volume allowing direct manipulation of elements.
    
    data = np.array(vol, copy=False)
    
    # Here we need to watch out since the typical
    # `data = data / data.max()`
    # will actually create a new array, not bound to the volume.
    # `out=data` makes sure we write to our original array.
    np.divide(data, np.max(data), out=data)


def downsample(vol, factor):
    """ Rough downsampling of the volume by an integer factor """

    assert isinstance(factor, int)

    # Get the data
    data = np.array(vol, copy=False)
    
    # Create a downsampled copy of the data
    data = data[::factor, ::factor, ::factor]

    # Create new volume
    out = pydeform.Volume(data)
    # Set metadata
    out.origin = vol.origin
    out.spacing = np.array(vol.spacing)*factor
    out.direction = vol.direction

    return out
