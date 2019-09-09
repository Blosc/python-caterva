import cat4py as cat
import numpy as np


pshape = (5, 7)
shape = (13, 20)

dtype = np.bool
itemsize = np.dtype(dtype).itemsize

# Create a buffer
nparray = np.random.choice(a=[True, False], size=np.prod(shape)).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray, pshape=pshape)

# Convert a caterva array to a numpy array
nparray2 = a.to_numpy(dtype)

np.testing.assert_almost_equal(nparray, nparray2)
