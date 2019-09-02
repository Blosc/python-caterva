import cat4py as cat
import zarr
import numpy as np
import os
from time import time


# Dimensions for big matrix
shape = (500, 1000)
pshape = (100, 100)

# filename = "bench_getitem.cat"
filename = None
if filename is not None and os.path.exists(filename):
    # Remove file on disk
    os.remove(filename)

dtype = np.float64
itemsize = np.dtype(dtype).itemsize

# Create an empty caterva array (on disk)
a = cat.empty(shape, pshape, itemsize=itemsize, filename=filename, compcode=0)
# Fill an empty caterva array using a block iterator
t0 = time()
nchunks = 0
for block, info in a.iter_write():
    nparray = np.arange(nchunks * info.size, (nchunks + 1) * info.size, dtype=dtype)
    block[:] = bytes(nparray)
    nchunks += 1
t1 = time()
print("Time for filling caterva array: %.3fs" % (t1 - t0))

t0 = time()
z = zarr.empty(shape, chunks=pshape, dtype=dtype)
for block, info in a.iter_read(pshape):
    z[info.slice] = np.frombuffer(block, dtype=dtype).reshape(pshape)
t1 = time()
print("Time for filling zarr array: %.3fs" % (t1 - t0))

# Check that the retrieved items are correct
t0 = time()
count = 0
for block, info in a.iter_read(pshape):
    nparray = np.arange(count, count + info.size, dtype=dtype)
    assert block == bytes(nparray)
    count += info.size
t1 = time()
print("Time for reading with iterator: %.3fs" % (t1 - t0))

# Time getitem with caterva
t0 = time()
for i in range(shape[0]):
    rbytes = a[i]
    block = np.frombuffer(rbytes, dtype=dtype).reshape((1, shape[1]))
    # print(block)
t1 = time()
print("Time for reading with getitem (caterva): %.3fs" % (t1 - t0))

# Time getitem with zarr
t0 = time()
for i in range(shape[0]):
    block = z[i]
    # print(block)
t1 = time()
print("Time for reading with getitem (zarr): %.3fs" % (t1 - t0))


if filename is not None:
    print("File is available at:", os.path.abspath(filename))
