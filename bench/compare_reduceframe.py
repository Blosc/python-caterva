# Benchmark for comparing reducing on-disk frames for
# multidimensional arrays using different methods:
# * Opening an on-disk frame without copying
# * Loading the frame in-memory

import caterva as cat
import numpy as np
import os
from time import time
import platform

macosx = 'Darwin' in platform.platform()
linux = 'Linux' in platform.platform()

# Dimensions, type and persistency properties for the arrays
shape = (100, 5000, 250)
chunkshape = (20, 100, 50)
blockshape = (10, 50, 25)

dtype = np.float64

# Compression properties
cname = "lz4"
clevel = 5
filter = cat.SHUFFLE
nthreads = 4

fname_npy = "compare_reduceframe.npy"
if os.path.exists(fname_npy):
    os.remove(fname_npy)
fname_cat = "compare_reduceframe.cat"
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

def bench_read_numpy(fname, copy):
    if macosx: os.system("/usr/sbin/purge")
    t0 = time()
    mmap_mode = None if copy else 'r'
    a = np.load(fname, mmap_mode=mmap_mode)
    t1 = time()
    print("Time for opening the on-disk frame (numpy, copy=%s): %.3fs" % (copy, (t1 - t0)))

    if macosx: os.system("/usr/sbin/purge")
    t0 = time()
    acc = a.sum()
    del a
    t1 = time()
    print("Time for reducing with (numpy, copy=%s): %.3fs" % (copy, (t1 - t0)))
    return acc

def bench_read_caterva(fname, copy):
    if macosx: os.system("/usr/sbin/purge")
    t0 = time()
    a = cat.open(fname, copy=copy)
    t1 = time()
    print("Time for opening the on-disk frame (caterva, copy=%s): %.3fs" % (copy, (t1 - t0)))

    if macosx: os.system("/usr/sbin/purge")
    t0 = time()
    acc = 0
    for (block, info) in a.iter_read():
        block = np.frombuffer(block, dtype=dtype).reshape(info.shape)
        acc += np.sum(block)
    del a
    t1 = time()
    print("Time for reducing with (caterva, copy=%s): %.3fs" % (copy, (t1 - t0)))
    return acc

acc_npy1 = bench_read_numpy(fname_npy, copy=False)
acc_npy2 = bench_read_numpy(fname_npy, copy=True)
np.testing.assert_allclose(acc_npy1, acc_npy2)
print()
acc_cat1 = bench_read_caterva(fname_cat, copy=False)
np.testing.assert_allclose(acc_cat1, acc_npy1)
acc_cat2 = bench_read_caterva(fname_cat, copy=True)
np.testing.assert_allclose(acc_cat1, acc_npy2)

os.remove(fname_npy)
os.remove(fname_cat)
