import cat4py as cat
import numpy as np


shape = (10, 10)
chunks = (5, 7)
blocks = (2, 2)

slices = (slice(2, 5), slice(4, 8))

dtype = np.int32
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.asarray(nparray, chunks=chunks, blocks=blocks)

# Get a slice
buffer = np.asarray(a[slices]).view(dtype)
buffer2 = nparray[slices]

np.testing.assert_almost_equal(buffer, buffer2)

a[slices] = np.ones((5, 5), dtype=dtype)

print(np.asarray(a[...]).view(dtype))
