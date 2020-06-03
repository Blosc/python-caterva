import cat4py as cat
import numpy as np
import os
from itertools import zip_longest as lzip


shape = (156, 223)
chunkshape = (22, 32)
blockshape = (12, 7)

filename = "ex_iters.cat"
if os.path.exists(filename):
    # Remove file on disk
    os.remove(filename)

dtype = np.complex128
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray)

# Create an empty caterva array (on disk)
b = cat.empty(shape, chunkshape=chunkshape, blockshape=blockshape,
              filename=filename, itemsize=itemsize)

# Fill an empty caterva array using a block iterator
for block, info in b.iter_write():
    block[:] = bytes(nparray[info.slice])

# Load file
c = cat.from_file(filename)

# Assert both caterva arrays
itershape = (5, 5)
for (block1, info1), (block2, info2) in lzip(a.iter_read(itershape), c.iter_read(itershape)):
    assert bytes(block1) == block2

# Remove file on disk
os.remove(filename)
