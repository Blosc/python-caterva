import cat4py as cat
import numpy as np
import os
from time import time

# Dimensions for big matrix
shape = (500, 500)
pshape = (50, 50)

# filename = "bench_getitem.cat"
filename = None
if filename is not None and os.path.exists(filename):
    # Remove file on disk
    os.remove(filename)

dtype = np.float64
itemsize = np.dtype(dtype).itemsize

# Create an empty caterva array (on disk)
a = cat.empty(shape, pshape, itemsize=itemsize, filename=filename)

# Fill an empty caterva array using a block iterator
t0 = time()
count = 0
for block, info in a.iter_write():
    nparray = np.arange(count, count + info.size, dtype=dtype)
    block[:] = bytes(nparray)
    count += info.size
t1 = time()
print("Time for filling: %.3fs" % (t1 - t0))

# Check that the retrieved items are correct
t0 = time()
count = 0
for block, info in a.iter_read(pshape):
    nparray = np.arange(count, count + info.size, dtype=dtype)
    assert block == bytes(nparray)
    count += info.size
t1 = time()
print("Time for reading with iterator: %.3fs" % (t1 - t0))

# Use getitem
t0 = time()
for i in range(shape[0]):
    rbytes = a[(slice(i), slice(None))]
    # print(np.frombuffer(rbytes, dtype=dtype))
t1 = time()
print("Time for reading with getitem: %.3fs" % (t1 - t0))


if filename is not None:
    print("File is available at:", os.path.abspath(filename))
