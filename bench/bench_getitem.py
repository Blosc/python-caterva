import cat4py as cat
import numpy as np
import os
from time import time


# Dimensions, type and persistency properties for the arrays
shape = (500, 1000)
pshape = (100, 100)
dtype = np.float64
persistent = False

if persistent:
    filename = "bench_getitem.cat"
    if os.path.exists(filename):
        # Remove file on disk
        os.remove(filename)
else:
    filename = None

itemsize = np.dtype(dtype).itemsize

# Create an empty caterva array
a = cat.empty(shape, pshape=pshape, dtype=dtype, filename=filename, compcode=0)

# Fill an empty caterva array using a block iterator
t0 = time()
count = 0
for block, info in a.iter_write():
    nparray = np.arange(count, count + info.size, dtype=dtype).reshape(info.shape)
    block[:] = nparray
    count += info.size
t1 = time()
print("Time for filling: %.3fs" % (t1 - t0))

# Check that the retrieved items are correct
t0 = time()
count = 0
for block, info in a.iter_read(pshape):
    nparray = np.arange(count, count + info.size, dtype=dtype).reshape(info.shape)
    np.testing.assert_allclose(block, nparray)
    count += info.size
t1 = time()
print("Time for reading with iterator: %.3fs" % (t1 - t0))

# Use getitem
t0 = time()
for i in range(shape[0]):
    rbytes = a[i]
    # print(np.frombuffer(rbytes, dtype=dtype).reshape((1, shape[1])))
t1 = time()
print("Time for reading with getitem: %.3fs" % (t1 - t0))


if persistent:
    print("File is available at:", os.path.abspath(filename))
