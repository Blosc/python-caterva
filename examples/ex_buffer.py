import cat4py as cat
import numpy as np


shape = (100, 77)
chunkshape = (13, 20)
blockshape = (10, 10)

itemsize = 4

# Create a buffer
buffer = bytes(np.prod(shape) * itemsize)

# Create a caterva array from a buffer
a = cat.from_buffer(buffer, shape, chunkshape=chunkshape, blockshape=blockshape, itemsize=itemsize)

# Convert a caterva array to a buffer
buffer2 = a.to_buffer()
assert buffer == buffer2
