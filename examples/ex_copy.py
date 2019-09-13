import cat4py as cat
import numpy as np


pshape = (5, 7)
shape = (13, 20)

itemsize = 4

# Create a buffer
buffer = bytes(np.prod(shape) * itemsize)

# Create a caterva array from a buffer
a = cat.from_buffer(buffer, shape, pshape=pshape, itemsize=itemsize)

# Get a copy of a caterva array
b = a.copy()

# Convert the copy to a buffer
buffer2 = b.to_buffer()

assert buffer == buffer2
