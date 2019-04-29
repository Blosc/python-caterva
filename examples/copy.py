import struct
import cat4py as cat
import numpy as np

cparams = cat.CParams(itemsize=4)  # we will be dealing with itemsizes of 4's
dparams = cat.DParams()

pshape = (5, 5)
shape = (10, 10)
# Create a cat container without partitions
a = cat.Container(pshape=pshape)

# Create a byte array
buf = bytes(np.arange(int(np.prod(shape)), dtype=np.float32))

# Fill th cat container array with the buffer
a.from_buffer(shape, buf)

b = a.copy()

c = b[2:6, 3:5]

d = np.frombuffer(c.to_buffer(), dtype=np.float32)

print(d)