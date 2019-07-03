import struct
import cat4py as cat
import numpy as np


pshape = (5, 5)
shape = (10, 10)

dtype = np.float32

itemsize = np.dtype(dtype).itemsize

a = cat.Container(pshape, itemsize=itemsize)

size = int(np.prod(shape))

buffer = np.arange(size, dtype=dtype).reshape(shape)

a.from_numpy(buffer)

buffer = buffer[2:5, 2:5]

c = a[2:5, 2:5]

print(c.size)
