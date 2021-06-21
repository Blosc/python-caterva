# Benchmark for comparing loading on-disk frames for
# multidimensional arrays using different methods:
# * Opening an on-disk frame without copying
# * Loading the frame in-memory

import caterva as cat
import numpy as np
import os
from time import time

# Dimensions, type and persistency properties for the arrays
shape = (100, 5000, 250)
chunkshape = (20, 100, 50)
blockshape = (10, 50, 25)

dtype = np.float64

# Compression properties
cname = "zstd"
clevel = 6
filter = cat.SHUFFLE
nthreads = 2

fname_npy = "compare_loadframe.npy"
if os.path.exists(fname_npy):
    os.remove(fname_npy)
fname_cat = "compare_loadframe.cat"
if os.path.exists(fname_cat):
    os.remove(fname_cat)

# Create content for populating arrays
t0 = time()
content = np.linspace(0, 10, int(np.prod(shape)), dtype=dtype).reshape(shape)
# content = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
t1 = time()
print("Time for filling array (numpy): %.3fs" % (t1 - t0))

t0 = time()
np.save(fname_npy, content)
t1 = time()
print("Time for storing array on-disk (numpy): %.3fs" % (t1 - t0))

# Create and fill a caterva array using a block iterator
t0 = time()
a = cat.empty(shape, chunkshape=chunkshape, blockshape=blockshape, itemsize=content.itemsize,
              filename=fname_cat,
              cname=cname, clevel=clevel, filters=[filter],
              nthreads=nthreads)
for block, info in a.iter_write():
    nparray = content[info.slice]
    block[:] = bytes(nparray)
acratio = a.cratio
del a
t1 = time()
print("Time for storing array on-disk (caterva, iter): %.3fs ; CRatio: %.1fx" % ((t1 - t0), acratio))

print()

# Setup the coordinates for random planes
planes_idx = np.random.randint(0, shape[1], 3)

def bench_read_numpy(fname, planes_idx, copy):
    t0 = time()
    mmap_mode = None if copy else 'r'
    a = np.load(fname, mmap_mode=mmap_mode)
    t1 = time()
    print("Time for opening the on-disk frame (numpy, copy=%s): %.3fs" % (copy, (t1 - t0)))

    t0 = time()
    for i in planes_idx:
        block = a[:, i, :]
        if not copy:
            # Do an actual read for memory mapped files
            # Do an actual read for memory mapped files
            block = block.copy()
    del a
    t1 = time()
    print("Time for reading with getitem (numpy, copy=%s): %.3fs" % (copy, (t1 - t0)))

def bench_read_caterva(fname, planes_idx, copy):
    t0 = time()
    a = cat.open(fname, copy=copy)
    t1 = time()
    print("Time for opening the on-disk frame (caterva, copy=%s): %.3fs" % (copy, (t1 - t0)))

    t0 = time()
    for i in planes_idx:
        rbytes = a[:, i, :]
        block = np.frombuffer(rbytes, dtype=dtype).reshape((shape[0], shape[2]))
    del a
    t1 = time()
    print("Time for reading with getitem (caterva, copy=%s): %.3fs" % (copy, (t1 - t0)))

bench_read_numpy(fname_npy, planes_idx, copy=False)
bench_read_numpy(fname_npy, planes_idx, copy=True)
print()
bench_read_caterva(fname_cat, planes_idx, copy=False)
bench_read_caterva(fname_cat, planes_idx, copy=True)

os.remove(fname_npy)
os.remove(fname_cat)
