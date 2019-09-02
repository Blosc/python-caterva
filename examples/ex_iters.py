import cat4py as cat
import numpy as np
import os
from itertools import zip_longest as lzip

pshape = (5, 5)
shape = (10, 10)
filename = "ex_iters.cat"
if os.path.exists(filename):
    # Remove file on disk
    os.remove(filename)

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
for block, info in b.iter_write():
    block[:] = bytes(nparray[info.slice])

# Assert both caterva arrays
for (block1, info1), (block2, info2) in lzip(a.iter_read(blockshape), b.iter_read(blockshape)):
    assert block1 == block2

print("File is available at:", os.path.abspath(filename))
