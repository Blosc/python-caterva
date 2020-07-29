import cat4py as cat
import numpy as np
import os


shape = (128, 128)
chunkshape = (32, 32)
blockshape = (8, 8)

filename = "ex_persistency.cat"
if os.path.exists(filename):
    # Remove file on disk
    os.remove(filename)

dtype = np.dtype(np.complex128)
itemsize = dtype.itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array (on disk)
a = cat.from_buffer(bytes(nparray), nparray.shape, chunkshape=chunkshape, blockshape=blockshape,
                    filename=filename, itemsize=itemsize)

# Read a caterva array from disk
b = cat.from_file(filename)

# Convert a caterva array to a numpy array
nparray2 = np.asarray(cat.from_buffer(b.to_buffer(), b.shape, b.itemsize, str(dtype)))

np.testing.assert_almost_equal(nparray, nparray2)

# Remove file on disk
os.remove(filename)
