import cat4py as cat
import numpy as np
import os


shape = (128, 128)
chunks = (32, 32)
blocks = (8, 8)

urlpath = "ex_persistency.cat"
if os.path.exists(urlpath):
    # Remove file on disk
    os.remove(urlpath)

dtype = np.dtype(np.complex128)
itemsize = dtype.itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array (on disk)
a = cat.from_buffer(bytes(nparray), nparray.shape, chunks=chunks, blocks=blocks,
                    urlpath=urlpath, itemsize=itemsize)

# Read a caterva array from disk
b = cat.open(urlpath)

# Convert a caterva array to a numpy array
nparray2 = np.asarray(cat.from_buffer(b.to_buffer(), b.shape, b.itemsize)).view(dtype)

np.testing.assert_almost_equal(nparray, nparray2)

# Remove file on disk
os.remove(urlpath)
