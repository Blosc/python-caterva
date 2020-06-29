import cat4py as cat
import numpy as np
import os
from itertools import zip_longest as lzip


shape = (10, 10)
chunkshape = (10, 10)
blockshape = (10, 10)

filename = "ex_iters.cat"
if os.path.exists(filename):
    # Remove file on disk
    os.remove(filename)

dtype = np.dtype(np.complex128)
itemsize = dtype.itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.asarray(nparray)

# Create an empty caterva array (on disk)
print(str(dtype))
b = cat.empty(shape, dtype=str(dtype), itemsize=itemsize, chunkshape=chunkshape, blockshape=blockshape,
              filename=filename)

# Fill an empty caterva array using a block iterator
for block, info in b.iter_write():
    block[:] = bytes(nparray[info.slice])

# Load file
c = cat.from_file(filename)

# Assert both caterva arrays
itershape = (5, 5)
for (block1, info1), (block2, info2) in lzip(a.iter_read(itershape), c.iter_read(itershape)):
    np.testing.assert_equal(np.asarray(block1), np.asarray(block2))

# Remove file on disk
os.remove(filename)
