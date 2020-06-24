import cat4py as cat
import numpy as np


shape = (10, 10)
chunkshape = (10, 10)
blockshape = (10, 10)

dtype = np.dtype(np.float64)

# Create a buffer
buffer = bytes(np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape))

# Create a caterva array from a buffer
a = cat.from_buffer(buffer, shape, chunkshape=chunkshape, blockshape=blockshape, itemsize=dtype.itemsize)

# Get a copy of a caterva array (plainbuffer)
b = a.copy()

c = memoryview(b)

# Convert the copy to a buffer
buffer1 = a.to_buffer()
buffer2 = b.to_buffer()

assert buffer1 == buffer2
