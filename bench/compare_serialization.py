# Benchmark for comparing serializing/deserializing frames for
# multidimensional arrays using different methods:
# * to_sframe() / from_sframe()
# * Numpy copy
# * PyArrow
# * Pickle v4
# * Pickle v5 (in the future)

import caterva as cat
import numpy as np
from time import time
import pyarrow as pa

import pickle

check_roundtrip = False  # set this to True to check for roundtrip validity

# Dimensions, type and persistency properties for the arrays
shape = (100, 5000, 250)
chunkshape = (20, 500, 100)
blockshape = (10, 50, 50)
dtype = "f8"

# Compression properties
cname = "lz4"
clevel = 3
# cname = "zstd"
# clevel = 1
filter = cat.SHUFFLE
nthreads = 4

# Create a plain numpy array
t0 = time()
arr = np.linspace(0, 10, int(np.prod(shape)), dtype=dtype).reshape(shape)
# arr = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
t1 = time()
print("Time for filling array (numpy): %.3fs" % (t1 - t0))

t0 = time()
arr2 = arr.copy()
t1 = time()
print("Time for copying array in-memory (numpy): %.3fs" % (t1 - t0))

# Create and fill a caterva array using a block iterator and an in-memory frame
t0 = time()
carr = cat.empty(shape, np.dtype(dtype).itemsize, dtype=dtype, chunkshape=chunkshape, blockshape=blockshape,
                 enforceframe=True,
                 cname=cname, clevel=clevel, filters=[filter],
                 cnthreads=nthreads, dnthreads=nthreads)
for block, info in carr.iter_write():
    nparray = arr[info.slice]
    block[:] = bytes(nparray)
acratio = carr.cratio
t1 = time()
print("Time for creating an array in-memory (numpy -> caterva, copy): %.3fs ; CRatio: %.1fx" % ((t1 - t0), acratio))

print()

t0 = time()
sframe_nocopy = carr.sframe
t1 = time()
print("Time for serializing array in-memory (caterva, no-copy): %.3fs" % (t1 - t0))

t0 = time()
sframe_copy = carr.to_sframe()
t1 = time()
print("Time for serializing array in-memory (caterva, copy): %.3fs" % (t1 - t0))

t0 = time()
serialized = pa.serialize(arr)
pyarrow_nocopy = serialized.to_components()
t1 = time()
print("Time for serializing array in-memory (arrow, no-copy): %.3fs" % (t1 - t0))

t0 = time()
pyarrow_copy = pa.serialize(arr).to_buffer().to_pybytes()
t1 = time()
print("Time for serializing array in-memory (arrow, copy): %.3fs" % (t1 - t0))

t0 = time()
frame_pickle = pickle.dumps(arr, protocol=4)
t1 = time()
print("Time for serializing array in-memory (pickle4, copy): %.3fs" % (t1 - t0))

t0 = time()
carr2 = cat.from_sframe(sframe_nocopy, copy=False)
t1 = time()
print("Time for de-serializing array in-memory (caterva, no-copy): %.3fs" % (t1 - t0))

if check_roundtrip:
    print("The roundtrip is... ", end="", flush=True)
    np.testing.assert_allclose(carr2, arr)
    print("ok!")

t0 = time()
arr2 = pa.deserialize_components(pyarrow_nocopy)
t1 = time()
print("Time for de-serializing array in-memory (arrow, no-copy): %.3fs" % (t1 - t0))

if check_roundtrip:
    print("The roundtrip is... ", end="", flush=True)
    np.testing.assert_allclose(arr2, arr)
    print("ok!")

t0 = time()
arr2 = pa.deserialize(pyarrow_copy)
t1 = time()
print("Time for de-serializing array in-memory (arrow, copy): %.3fs" % (t1 - t0))

if check_roundtrip:
    print("The roundtrip is... ", end="", flush=True)
    np.testing.assert_allclose(arr2, arr)
    print("ok!")

t0 = time()
arr2 = pickle.loads(frame_pickle)
t1 = time()
print("Time for de-serializing array in-memory (pickle4, copy): %.3fs" % (t1 - t0))

if check_roundtrip:
    print("The roundtrip is... ", end="", flush=True)
    np.testing.assert_allclose(arr2, arr)
    print("ok!")

print()
t0 = time()
for i in range(1):
    carr3 = cat.from_sframe(sframe_copy)
    arr2 = np.asarray(carr3.copy())
t1 = time()
print("Time for re-creating array in-memory (caterva -> numpy, copy): %.3fs" % (t1 - t0))

if check_roundtrip:
    print("The roundtrip is... ", end="", flush=True)
    np.testing.assert_allclose(arr2, arr)
    print("ok!")

print()
arrsize = arr.size * arr.itemsize
time_100Mbps = arrsize / (10 * 2 ** 20)
print("Time to transmit array at 100 Mbps (no compression):\t%6.3fs" % time_100Mbps)
ctime_100Mbps = (arrsize / acratio) / (10 * 2**20)
print("Time to transmit array at 100 Mbps (compression):\t%6.3fs" % ctime_100Mbps)
time_1Gbps = arrsize / (100 * 2 ** 20)
print("Time to transmit array at 1 Gbps (no compression):\t%6.3fs" % time_1Gbps)
ctime_1Gbps = (arrsize / acratio) / (100 * 2**20)
print("Time to transmit array at 1 Gbps (compression):\t\t%6.3fs" % ctime_1Gbps)
time_10Gbps = arrsize / (1000 * 2 ** 20)
print("Time to transmit array at 10 Gbps (no compression):\t%6.3fs" % time_10Gbps)
ctime_10Gbps = (arrsize / acratio) / (1000 * 2**20)
print("Time to transmit array at 10 Gbps (compression):\t%6.3fs" % ctime_10Gbps)
