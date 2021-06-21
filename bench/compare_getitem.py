import caterva as cat
import numpy as np
import os
import sys
from time import time


# Dimensions, type and persistency properties for the arrays
shape = (1000 * 1000,)
chunkshape = (100,)
blockshape = (25,)

dtype = np.float64
persistent = bool(sys.argv[1]) if len(sys.argv) > 1 else False

if persistent:
    filename = "bench_getitem.cat"
    if os.path.exists(filename):
        # Remove file on disk
        os.remove(filename)
else:
    filename = None

itemsize = np.dtype(dtype).itemsize

# Create an empty caterva array
a = cat.empty(shape, itemsize, dtype=str(np.dtype(dtype)), chunkshape=chunkshape, blockshape=blockshape,
              filename=filename, compcode=0)

# Fill an empty caterva array using a block iterator
t0 = time()
count = 0
for block, info in a.iter_write():
    nparray = np.arange(count, count + info.nitems, dtype=dtype).reshape(info.shape)
    block[:] = bytes(nparray)
    count += info.nitems
t1 = time()
print("Time for filling: %.3fs" % (t1 - t0))

# Check that the retrieved items are correct
t0 = time()
for block, info in a.iter_read(chunkshape):
    pass
t1 = time()
print("Time for reading with iterator: %.3fs" % (t1 - t0))

# Asserting results
count = 0
for block, info in a.iter_read(chunkshape):
    nparray = np.arange(count, count + info.nitems, dtype=dtype).reshape(info.shape)
    np.testing.assert_allclose(block, nparray)
    count += info.nitems

# Use getitem
t0 = time()
for i in range(shape[0] // chunkshape[0]):
    _ = a[i * 100: (i+1) * 100]
t1 = time()
print("Time for reading with getitem: %.3fs" % (t1 - t0))

count = 0
for i in range(shape[0] // chunkshape[0]):
    nparray = np.arange(count, count + chunkshape[0], dtype=dtype).reshape(chunkshape)
    np.testing.assert_allclose(a[i * chunkshape[0]: (i+1) * chunkshape[0]], nparray)
    count += chunkshape[0]


if persistent:
    os.remove(filename)
