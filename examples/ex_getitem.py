import cat4py as cat
import numpy as np


shape = (200, 132)
chunkshape = (55, 23)
blockshape = (5, 11)

slices = (5, ..., slice(2, 13))

dtype = np.float64
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.asarray(nparray, chunkshape=chunkshape, blockshape=blockshape)

# Get a slice
buffer = np.asarray(a[slices])
buffer2 = nparray[slices]

np.testing.assert_almost_equal(buffer, buffer2)
