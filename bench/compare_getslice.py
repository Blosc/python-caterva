# Benchmark for comparing speeds of getitem of hyperplanes on a
# multidimensional array and using different backends:
# Caterva, zarr and HDF5
# In brief, each approach has its own strengths and weaknesses.
#
# Usage: pass any argument for testing the persistent backends.
# Else, only in-memory containers will be tested.

import caterva as cat
import zarr
import numcodecs
import tables
import numpy as np
import os
import sys
import shutil
from time import time

persistent = bool(sys.argv[1]) if len(sys.argv) > 1 else False
if persistent:
    print("Testing the persistent backends...")
else:
    print("Testing the in-memory backends...")

# Dimensions and type properties for the arrays
# 'Small' arrays config follows...
shape = (100, 5000, 250)
chunkshape = (20, 500, 50)
blockshape = (10, 100, 25)
# This config generates containers of more than 2 GB in size
# shape = (250, 4000, 350)
# pshape = (200, 100, 100)
dtype = np.float64

# Compression properties
cname = "zstd"
clevel = 6
filter = cat.SHUFFLE
nthreads = 1
blocksize = int(np.prod(blockshape))

fname_cat = None
fname_zarr = None
fname_h5 = "whatever.h5"
if persistent:
    fname_cat = "compare_getslice.cat"
    if os.path.exists(fname_cat):
        os.remove(fname_cat)
    fname_zarr = "compare_getslice.zarr"
    if os.path.exists(fname_zarr):
        shutil.rmtree(fname_zarr)
    fname_h5 = "compare_getslice.h5"
    if os.path.exists(fname_h5):
        os.remove(fname_h5)

# Create content for populating arrays
content = np.random.normal(0, 1, int(np.prod(shape))).reshape(shape)

# Create and fill a caterva array using a buffer
# t0 = time()
# a = cat.from_buffer(bytes(content), shape, pshape=pshape, itemsize=content.itemsize, filename=fname_cat,
#                     cname=cname, clevel=clevel, filters=[filter],
#                     cnthreads=nthreads, dnthreads=nthreads)
# if persistent:
#     del a
# t1 = time()
# print("Time for filling array (caterva, from_buffer): %.3fs" % (t1 - t0))

# if fname_cat is not None and os.path.exists(fname_cat):
#     os.remove(fname_cat)

# Create and fill a caterva array using a block iterator
t0 = time()
a = cat.empty(shape, content.itemsize, chunkshape=chunkshape, blockshape=blockshape,
              dtype=str(content.dtype), filename=fname_cat,
              cname=cname, clevel=clevel, filters=[filter], nthreads=nthreads)
for block, info in a.iter_write():
    block[:] = bytes(content[info.slice])
acratio = a.cratio
if persistent:
    del a
t1 = time()
print("Time for filling array (caterva, iter): %.3fs ; CRatio: %.1fx" % ((t1 - t0), acratio))

# Create and fill a zarr array
t0 = time()
compressor = numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=filter, blocksize=blocksize)
numcodecs.blosc.set_nthreads(nthreads)
if persistent:
    z = zarr.open(fname_zarr, mode='w', shape=shape, chunks=chunkshape, dtype=dtype, compressor=compressor)
else:
    z = zarr.empty(shape=shape, chunks=chunkshape, dtype=dtype, compressor=compressor)
z[:] = content
zratio = z.nbytes / z.nbytes_stored
if persistent:
    del z
t1 = time()
print("Time for filling array (zarr): %.3fs ; CRatio: %.1fx" % ((t1 - t0), zratio))

# Create and fill a hdf5 array
t0 = time()
filters = tables.Filters(complevel=clevel, complib="blosc:%s" % cname, shuffle=True)
tables.set_blosc_max_threads(nthreads)
if persistent:
    h5f = tables.open_file(fname_h5, 'w')
else:
    h5f = tables.open_file(fname_h5, 'w', driver='H5FD_CORE', driver_core_backing_store=0)
h5ca = h5f.create_carray(h5f.root, 'carray', filters=filters, chunkshape=chunkshape, obj=content)
h5f.flush()
h5ratio = h5ca.size_in_memory / h5ca.size_on_disk
if persistent:
    h5f.close()
t1 = time()
print("Time for filling array (hdf5): %.3fs ; CRatio: %.1fx" % ((t1 - t0), h5ratio))

# Check that the contents are the same
t0 = time()
if persistent:
    a = cat.open(fname_cat, copy=False)  # reopen
    z = zarr.open(fname_zarr, mode='r')
    h5f = tables.open_file(fname_h5, 'r', filters=filters)
    h5ca = h5f.root.carray
for block, info in a.iter_read(chunkshape):
    block_cat = block
    block_zarr = z[info.slice]
    np.testing.assert_array_almost_equal(block_cat, block_zarr)
    block_h5 = h5ca[info.slice]
    np.testing.assert_array_almost_equal(block_cat, block_h5)
if persistent:
    del a
    del z
    h5f.close()
t1 = time()
print("Time for checking contents: %.3fs" % (t1 - t0))

# Setup the coordinates for random planes
planes_idx = np.random.randint(0, shape[1], 100)

# Time getitem with caterva
t0 = time()
if persistent:
    a = cat.open(fname_cat, copy=False)  # reopen
for i in planes_idx:
    rbytes = a[:, i, :]
del a
t1 = time()
print("Time for reading with getitem (caterva): %.3fs" % (t1 - t0))

# Time getitem with zarr
t0 = time()
if persistent:
    z = zarr.open(fname_zarr, mode='r')
for i in planes_idx:
    block = z[:, i, :]
del z
t1 = time()
print("Time for reading with getitem (zarr): %.3fs" % (t1 - t0))

# Time getitem with hdf5
t0 = time()
if persistent:
    h5f = tables.open_file(fname_h5, 'r', filters=filters)
h5ca = h5f.root.carray
for i in planes_idx:
    block = h5ca[:, i, :]
h5f.close()
t1 = time()
print("Time for reading with getitem (hdf5): %.3fs" % (t1 - t0))


if persistent:
    os.remove(fname_cat)
    shutil.rmtree(fname_zarr)
    os.remove(fname_h5)
