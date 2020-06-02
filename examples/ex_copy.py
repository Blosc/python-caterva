import cat4py as cat
import numpy as np


shape = (50, 123)
chunkshape = (13, 44)
blockshape = (5, 15)

itemsize = 8

# Create a buffer
buffer = bytes(np.prod(shape) * itemsize)

# Create a caterva array from a buffer
a = cat.from_buffer(buffer, shape, chunkshape=chunkshape, blockshape=blockshape, itemsize=itemsize)

# Get a copy of a caterva array (plainbuffer)
b = a.copy()

# Convert the copy to a buffer
buffer1 = a.to_buffer()
buffer2 = b.to_buffer()

assert buffer1 == buffer2
