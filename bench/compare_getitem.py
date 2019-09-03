import cat4py as cat
import zarr
from numcodecs import Blosc
import tables
import numpy as np
import os
import shutil
from time import time


# Dimensions, type and persistency properties for the arrays
shape = (500, 10000)
pshape = (50, 1000)
dtype = np.float64
persistent = False

# Compression properties
cname = "lz4"
compcode = 1  # keep in sync with above
clevel = 5
filter = Blosc.SHUFFLE

fname_cat = None
fname_zarr = None
fname_h5 = "whatever.h5"
if persistent:
    fname_cat = "compare_getitem.cat"
    if os.path.exists(fname_cat):
        os.remove(fname_cat)
    fname_zarr = "compare_getitem.zarr"
    if os.path.exists(fname_zarr):
        shutil.rmtree(fname_zarr)
    fname_h5 = "compare_getitem.h5"
    if os.path.exists(fname_h5):
        os.remove(fname_h5)

# Create content for populating arrays
content = np.linspace(0, 10, shape[0] * shape[1], dtype=dtype).reshape(shape)

itemsize = np.dtype(dtype).itemsize

# Create and fill a caterva array using a buffer
t0 = time()
a = cat.from_buffer(bytes(content), shape, pshape, itemsize=itemsize, filename=fname_cat,
                    compcode=compcode, clevel=clevel, filters=[filter])
t1 = time()
print("Time for filling array (caterva, from_buffer): %.3fs" % (t1 - t0))

# # Create and fill a caterva array using a block iterator
# if fname_cat is not None and os.path.exists(fname_cat):
#     os.remove(fname_cat)
# t0 = time()
# itemsize = np.dtype(dtype).itemsize
# a = cat.empty(shape, pshape, itemsize=itemsize, filename=fname_cat,
#               compcode=compcode, clevel=clevel, filters=[filter])
# for block, info in a.iter_write():
#     nparray = content[info.slice]
#     block[:] = bytes(nparray)
# t1 = time()
# print("Time for filling array (caterva, iter): %.3fs" % (t1 - t0))

# Create and fill a zarr array
t0 = time()
compressor = Blosc(cname=cname, clevel=clevel, shuffle=Blosc.SHUFFLE)
if persistent:
    z = zarr.open(fname_zarr, mode='w', shape=shape, chunks=pshape, dtype=dtype)
else:
    z = zarr.empty(shape=shape, chunks=pshape, dtype=dtype)
z[:] = content
t1 = time()
print("Time for filling array (zarr): %.3fs" % (t1 - t0))

# Create and fill a hdf5 array
t0 = time()
filters = tables.Filters(complevel=clevel, complib="blosc:%s" % cname, shuffle=True)
if persistent:
    h5f = tables.open_file(fname_h5, 'w', filters=filters)
else:
    h5f = tables.open_file(fname_h5, 'w', filters=filters, driver='H5FD_CORE')
h5ca = h5f.create_carray(h5f.root, 'carray', chunkshape=pshape, obj=content)
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

# Time getitem with caterva
t0 = time()
for i in range(shape[0]):
    rbytes = a[i]
    block = np.frombuffer(rbytes, dtype=dtype).reshape((1, shape[1]))
t1 = time()
print("Time for reading with getitem (caterva): %.3fs" % (t1 - t0))

# Time getitem with zarr
t0 = time()
for i in range(shape[0]):
    block = z[(i,)]
t1 = time()
print("Time for reading with getitem (zarr): %.3fs" % (t1 - t0))

# Time getitem with hdf5
t0 = time()
if persistent:
    h5f.close()
    h5f = tables.open_file(fname_h5, 'r', filters=filters)
h5ca = h5f.root.carray
for i in range(shape[0]):
    block = h5ca[i]
h5f.close()
t1 = time()
print("Time for reading with getitem (hdf5): %.3fs" % (t1 - t0))


if persistent:
    print("File for caterva is available at:", os.path.abspath(fname_cat))
    print("Storage for zarr is available at:", os.path.abspath(fname_zarr))
    print("File for hdf5 is available at:", os.path.abspath(fname_h5))
