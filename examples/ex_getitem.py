import cat4py as cat
import numpy as np


shape = (13, 20)
slices = (slice(3, 10), slice(5, 13))

dtype = np.float64
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray)

# Get a slice
buffer = a[slices]
buffer2 = bytes(nparray[slices])

assert buffer == buffer2
