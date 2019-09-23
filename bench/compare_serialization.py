# Benchmark for comparing serializing/deserializing frames for
# multidimensional arrays using different methods:
# * to_sframe() / from_sframe()
# * Numpy copy
# * PyArrow
# * Pickle v4
# * Pickle v5 (in the future)

import cat4py as cat
import numpy as np
from time import time
import pyarrow as pa

import pickle
assert(pickle.HIGHEST_PROTOCOL <= 4)

check_roundtrip = True  # set this to True to check for roundtrip validity

# Dimensions, type and persistency properties for the arrays
shape = (100, 5000, 250)
# pshape = (20, 500, 50)
pshape = (1, 5000, 250)
dtype = "float64"

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
carr = cat.empty(shape, pshape=pshape, itemsize=arr.itemsize, memframe=True,
                 cname=cname, clevel=clevel, filters=[filter],
                 cnthreads=nthreads, dnthreads=nthreads,
                 metalayers={"numpy": dtype})
for block, info in carr.iter_write():
    nparray = arr[info.slice]
    block[:] = bytes(nparray)
acratio = carr.cratio
t1 = time()
print("Time for creating an array in-memory (caterva, iter): %.3fs ; CRatio: %.1fx" % ((t1 - t0), acratio))

print()

t0 = time()
sframe = carr.to_sframe()
t1 = time()
print("Time for serializing array in-memory (caterva, copy): %.3fs" % (t1 - t0))

t0 = time()
serialized = pa.serialize(arr)
components = serialized.to_components()
t1 = time()
print("Time for serializing array in-memory (arrow, no-copy): %.3fs" % (t1 - t0))

t0 = time()
frame_pickle = pickle.dumps(arr, protocol=4)
t1 = time()
print("Time for serializing array in-memory (pickle4, copy): %.3fs" % (t1 - t0))

t0 = time()
carr2 = cat.from_sframe(sframe)
t1 = time()
print("Time for de-serializing array in-memory (caterva, no-copy): %.3fs" % (t1 - t0))

# Activate this when we would have a proper NPArray class with an __array__ method
# if check_roundtrip:
#     print("Checking that the roundtrip is... ", end="")
#     np.testing.assert_allclose(carr2, arr)
#     print("ok!")

t0 = time()
arr2 = pa.deserialize_components(components)
t1 = time()
print("Time for de-serializing array in-memory (arrow, no-copy): %.3fs" % (t1 - t0))

if check_roundtrip:
    print("Checking that the roundtrip is... ", end="")
    np.testing.assert_allclose(arr2, arr)
    print("ok!")

t0 = time()
arr2 = pickle.loads(frame_pickle)
t1 = time()
print("Time for de-serializing array in-memory (pickle4, copy): %.3fs" % (t1 - t0))

if check_roundtrip:
    print("Checking that the roundtrip is... ", end="")
    np.testing.assert_allclose(arr2, arr)
    print("ok!")

t0 = time()
arr2 = pa.deserialize_components(components).copy()
t1 = time()
print("Time for de-serializing array in-memory (arrow, copy): %.3fs" % (t1 - t0))

if check_roundtrip:
    print("Checking that the roundtrip is... ", end="")
    np.testing.assert_allclose(arr2, arr)
    print("ok!")

t0 = time()
for i in range(1):
    carr3 = cat.from_sframe(sframe)
    dtype_deserialized = carr3.get_metalayer("numpy")
    arr2 = carr3.to_numpy(dtype=dtype_deserialized)
t1 = time()
print("Time for re-creating array in-memory (caterva -> numpy, copy): %.3fs" % (t1 - t0))

if check_roundtrip:
    print("Checking that the roundtrip is... ", end="")
    np.testing.assert_allclose(arr2, arr)
    print("ok!")
