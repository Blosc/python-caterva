import cat4py as cat
import numpy as np
import os


pshape = (5, 7)
shape = (13, 20)
filename = "ex_persistency.cat"
if os.path.exists(filename):
    # Remove file on disk
    os.remove(filename)

dtype = np.complex128
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array (on disk)
a = cat.from_numpy(nparray, pshape, filename, itemsize=itemsize)

# Read a caterva array from disk
b = cat.from_file(filename)

# Convert a caterva array to a numpy array
nparray2 = b.to_numpy(dtype=dtype)

np.testing.assert_almost_equal(nparray, nparray2)

print("File is available at:", os.path.abspath(filename))
