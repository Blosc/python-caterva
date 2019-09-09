import cat4py as cat
import numpy as np


pshape = (5, 7)
shape = (13, 20)

itemsize = 4

# Create a buffer
buffer = bytes(np.prod(shape) * itemsize)

# Create a caterva array from a buffer
a = cat.from_buffer(buffer, shape, pshape=pshape, itemsize=itemsize)

# Convert a caterva array to a buffer
buffer2 = a.to_buffer()

assert buffer == buffer2
