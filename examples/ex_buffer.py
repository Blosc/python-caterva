import cat4py as cat
import numpy as np

np.random.seed(123)

shape = (50, 50)
chunkshape = (49, 49)
blockshape = (48, 48)

itemsize = 8

# Create a buffer
buffer = bytes(np.random.normal(0, 1, np.prod(shape)) * itemsize)

# Create a caterva array from a buffer

a = cat.from_buffer(buffer, shape, chunkshape=chunkshape, blockshape=blockshape, itemsize=itemsize)

print(a.cratio)
# Convert a caterva array to a buffer
buffer2 = a.to_buffer()
assert buffer == buffer2
