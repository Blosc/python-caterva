import cat4py as cat
import numpy as np
import os
from itertools import zip_longest as lzip

pshape = (5, 5)
shape = (10, 10)
filename = "meta-array.cat"

blockshape = (5, 5)

dtype = np.complex128
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray, pshape, itemsize=itemsize)

# Create an empty caterva array (on disk)
b = cat.empty(shape, pshape, filename, itemsize=itemsize, metalayers={"numpy": {"dtype": "int32"}})

assert(b.has_metalayer("numpy") is True)

assert(b.get_metalayer("numpy") == {b"dtype": b"int32"})

# Fill an empty caterva array using a block iterator
for block, info in b.iter_write(dtype):
    block[:] = nparray[info.slice]

# Remove file on disk
os.remove(filename)
