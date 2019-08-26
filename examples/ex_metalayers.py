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
b = cat.empty(shape, pshape, filename, itemsize=itemsize)

# Fill an empty caterva array using a block iterator
for block, info in b.iter_write(dtype):
    block[:] = nparray[info.slice]

assert(b.has_metalayer("specs") is False)

data = {b"name": b"array_1", b"id": 12345678}
b.add_metalayer("specs", data)

assert(b.has_metalayer("specs") is True)

assert(b.get_metalayer("specs") == data)

data = {b"name": b"array_2", b"id": 12345678}
b.update_metalayer("specs", data)

assert(b.get_metalayer("specs") == data)

# Remove file on disk
os.remove(filename)
