# Benchmark for comparing speeds of getitem of hyperplanes on a
# multidimensional array and using different backends:
# Caterva, zarr and HDF5
# In brief, each approach has its own strengths and weaknesses.

import cat4py as cat
import zarr
import numcodecs
import tables
import numpy as np
import os
import shutil
from time import time

persistent = False   # set this to True to benchmark the persistent storage for the backends

# Dimensions, type and persistency properties for the arrays
shape = (50, 5000, 100)
pshape = (10, 50, 20)
dtype = np.float64

# Compression properties
cname = "lz4"
compcode = cat.LZ4  # keep in sync with above
clevel = 5
filter = cat.SHUFFLE

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
content = np.linspace(0, 10, int(np.prod(shape)), dtype=dtype).reshape(shape)

itemsize = np.dtype(dtype).itemsize

# Create and fill a caterva array using a buffer
t0 = time()
a = cat.from_buffer(bytes(content), shape, pshape=pshape, itemsize=itemsize, filename=fname_cat,
                    compcode=compcode, clevel=clevel, filters=[filter])
t1 = time()
print("Time for filling array (caterva, from_buffer): %.3fs" % (t1 - t0))

# # Create and fill a caterva array using a block iterator
# if fname_cat is not None and os.path.exists(fname_cat):
#     os.remove(fname_cat)
# t0 = time()
# itemsize = np.dtype(dtype).itemsize
# a = cat.empty(shape, pshape=pshape, itemsize=itemsize, filename=fname_cat,
#               compcode=compcode, clevel=clevel, filters=[filter])
# for block, info in a.iter_write():
#     nparray = content[info.slice]
#     block[:] = bytes(nparray)
# t1 = time()
# print("Time for filling array (caterva, iter): %.3fs" % (t1 - t0))

# Create and fill a zarr array
t0 = time()
compressor = numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=filter)
if persistent:
    z = zarr.open(fname_zarr, mode='w', shape=shape, chunks=pshape, dtype=dtype, compressor=compressor)
else:
    z = zarr.empty(shape=shape, chunks=pshape, dtype=dtype, compressor=compressor)
z[:] = content
t1 = time()
print("Time for filling array (zarr): %.3fs" % (t1 - t0))

# Create and fill a hdf5 array
t0 = time()
filters = tables.Filters(complevel=clevel, complib="blosc:%s" % cname, shuffle=True)
if persistent:
    h5f = tables.open_file(fname_h5, 'w')
else:
    h5f = tables.open_file(fname_h5, 'w', driver='H5FD_CORE', driver_core_backing_store=0)
h5ca = h5f.create_carray(h5f.root, 'carray', filters=filters, chunkshape=pshape, obj=content)
h5f.flush()
t1 = time()
print("Time for filling array (hdf5): %.3fs" % (t1 - t0))

# Check that the contents are the same
t0 = time()
for block, info in a.iter_read(pshape):
    block_cat = np.frombuffer(block, dtype=dtype).reshape(pshape)
    block_zarr = z[info.slice]
    np.testing.assert_array_almost_equal(block_cat, block_zarr)
    block_h5 = h5ca[info.slice]
    np.testing.assert_array_almost_equal(block_cat, block_h5)
t1 = time()
print("Time for checking contents: %.3fs" % (t1 - t0))

# Get the coordinates for random planes
planes_idx = np.random.randint(0, shape[1], 100)

# Time getitem with caterva
t0 = time()
for i in planes_idx:
    rbytes = a[:,i,:]
    block = np.frombuffer(rbytes, dtype=dtype).reshape((shape[0], shape[2]))
t1 = time()
print("Time for reading with getitem (caterva): %.3fs" % (t1 - t0))

# Time getitem with zarr
t0 = time()
for i in planes_idx:
    block = z[:,i,:]
t1 = time()
print("Time for reading with getitem (zarr): %.3fs" % (t1 - t0))

# Time getitem with hdf5
t0 = time()
if persistent:
    h5f.close()
    h5f = tables.open_file(fname_h5, 'r', filters=filters)
h5ca = h5f.root.carray
for i in planes_idx:
    block = h5ca[:,i,:]
h5f.close()
t1 = time()
print("Time for reading with getitem (hdf5): %.3fs" % (t1 - t0))


if persistent:
    print("File for caterva is available at:", os.path.abspath(fname_cat))
    print("Storage for zarr is available at:", os.path.abspath(fname_zarr))
    print("File for hdf5 is available at:", os.path.abspath(fname_h5))
