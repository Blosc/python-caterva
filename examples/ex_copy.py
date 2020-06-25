import cat4py as cat
import numpy as np


shape = (10, 10)
chunkshape = (10, 10)
blockshape = (10, 10)

dtype = np.dtype(np.float64)

# Create a buffer
buffer = bytes(np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape))

# Create a caterva array from a buffer
a = cat.from_buffer(buffer, shape, dtype=dtype.type, chunkshape=chunkshape, blockshape=blockshape, itemsize=dtype.itemsize)

# Get a copy of a caterva array (plainbuffer)
b = a.copy()
d = b.copy()

aux = b.to_numpy()
aux[1, 2] = 0
aux2 = cat.asarray(aux)

print(np.asarray(aux2))

c = np.asarray(b)

c[3:5, 2:7] = 0
print(c)

del b

print(c)

# Convert the copy to a buffer
buffer1 = a.to_buffer()
buffer2 = d.to_buffer()

assert buffer1 == buffer2

