import cat4py as cat
import numpy as np


shape = (1234, 23)
chunks = (253, 23)
blocks = (10, 23)

dtype = bool

# Create a buffer
nparray = np.random.choice(a=[True, False], size=np.prod(shape)).reshape(shape)

# Create a caterva array from a numpy array
a = cat.asarray(nparray, chunks=chunks, blocks=blocks)
b = a.copy()
# Convert a caterva array to a numpy array
nparray2 = np.asarray(b).view(dtype)


np.testing.assert_almost_equal(nparray, nparray2)
