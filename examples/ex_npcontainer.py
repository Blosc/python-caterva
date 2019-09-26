import cat4py as cat
import numpy as np
import os
from itertools import zip_longest as lzip

pshape = (5, 5)
shape = (10, 10)
blockshape = (2, 3)

dtype = np.float64
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.npcontainer.from_numpy(nparray, pshape=pshape)
print(a.to_numpy())

# Create an empty caterva array (on disk)
b = cat.npcontainer.empty(shape, dtype=dtype, itemsize=itemsize)

# Fill an empty caterva array using a block iterator
for block, info in b.iter_write():
    block[:] = nparray[info.slice]

# Assert both caterva arrays
for (block1, info1), (block2, info2) in lzip(a.iter_read(blockshape), b.iter_read(blockshape)):
    np.testing.assert_almost_equal(block1, block2)

b[2:6, 4:6] = np.ones((4, 2), dtype=dtype)

print(b.to_numpy())

print(b[5:10, 5:10])

