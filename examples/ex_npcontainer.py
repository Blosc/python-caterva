import os
from itertools import zip_longest as lzip
import cat4py as cat
import numpy as np

shape = (512, 512)
chunkshape = (121, 99)
blockshape = (12, 31)

persistent = True
if persistent:
    filename = "ex_nparray.cat"
    if os.path.exists(filename):
        os.remove(filename)
else:
    filename = None

dtype = np.float64
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.linspace(0, 1, int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray, chunkshape=chunkshape, blockshape=blockshape,
                   enforceframe=persistent, filename=filename)

# Create an empty caterva array (on disk)
b = cat.empty(shape, dtype=dtype, itemsize=itemsize)

# Fill an empty caterva array using a block iterator
for block, info in b.iter_write():
    block[:] = nparray[info.slice]

# Assert both caterva arrays
for (block1, info1), (block2, info2) in lzip(a.iter_read(blockshape), b.iter_read(blockshape)):
    np.testing.assert_almost_equal(block1, block2)

print(b[5:10, 5:10])

if persistent:
    os.remove(filename)
