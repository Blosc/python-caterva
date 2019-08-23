import cat4py as cat
import numpy as np
import os
from itertools import zip_longest as lzip

pshape = (6, 7)
shape = (13, 20)
filename = "iters-array.cat"

blockshape = (3, 2)

dtype = np.complex128
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray, pshape, itemsize=itemsize)

# Create an empty caterva array (on disk)
b = cat.empty(shape, pshape, filename, itemsize=itemsize)

print(b.has_metalayer("numpy"))
print(b.add_metalayer("numpy", b"hola"))
print(b.has_metalayer("numpy"))
print(b.get_metalayer("numpy"))


# Fill an empty caterva array using a block iterator
for block, info in b.iter_write(dtype):
    block[:] = nparray[info.slice]

# Assert both caterva arrays
for (block1, info1), (block2, info2) in lzip(a.iter_read(blockshape, dtype), b.iter_read(blockshape, dtype)):
    np.testing.assert_almost_equal(block1, block2)

# Remove file on disk
os.remove(filename)
