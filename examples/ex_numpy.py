import cat4py as cat
import numpy as np


shape = (1234, 23)
chunkshape = (253, 23)
blockshape = (10, 23)

dtype = np.bool

# Create a buffer
nparray = np.random.choice(a=[True, False], size=np.prod(shape)).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray, chunkshape=chunkshape, blockshape=blockshape)

# Convert a caterva array to a numpy array
nparray2 = a.to_numpy()

np.testing.assert_almost_equal(nparray, nparray2)
