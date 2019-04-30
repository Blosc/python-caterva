import struct
import cat4py as cat
import numpy as np

cparams = cat.CParams(itemsize=4)  # we will be dealing with itemsizes of 4's
dparams = cat.DParams()

pshape = (5, 5)
shape = (10, 10)

# Create a cat container with partitions
a = cat.Container(pshape=pshape)
buf = bytes(np.arange(int(np.prod(shape)), dtype=np.float32))
a.from_buffer(shape, buf)

# Create a copy to a plain buffer
b1 = a.copy()
c = np.frombuffer(b1.to_buffer(), dtype=np.float32)
print(f"schunk to plainbuffer")
print(c)

# Create a copy to a schunk
b2 = a.copy(pshape=(3, 3), cparams=cat.CParams(itemsize=4, compcode=0, filters=2))
c = np.frombuffer(b2.to_buffer(), dtype=np.float32)
print(f"schunk to schunk")
print(c)

# Create a cat container without partitions
a = cat.Container()
buf = bytes(np.arange(int(np.prod(shape)), dtype=np.float32))
a.from_buffer(shape, buf)

# Create a copy to a plain buffer
b1 = a.copy()
c = np.frombuffer(b1.to_buffer(), dtype=np.float32)
print(f"schunk to plainbuffer")
print(c)

# Create a copy to a schunk
b2 = a.copy(pshape=(5, 5), cparams=cat.CParams(itemsize=4, compcode=1))
c = np.frombuffer(b2.to_buffer(), dtype=np.float32)
print(f"schunk to schunk")
print(c)
