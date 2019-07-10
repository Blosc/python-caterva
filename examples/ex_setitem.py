import cat4py as cat
import numpy as np


shape = (13, 20)
slices = (slice(3, 10), slice(5, 13))

dtype = np.float64
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray, itemsize=itemsize)

# Set a slice
slices_shape = [s.stop - s.start for s in slices]
a[slices] = np.zeros(slices_shape, dtype=dtype)
nparray[slices] = np.zeros(slices_shape, dtype=dtype)

# Convert a caterva array to a buffer
nparray2 = a.to_numpy(dtype)

np.testing.assert_almost_equal(nparray, nparray2)
