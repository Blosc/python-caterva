import struct
import cat4py as cat
import numpy as np

cparams = cat.CParams(itemsize=4)  # we will be dealing with itemsizes of 4's
dparams = cat.DParams()

pshape = (5, 5)
shape = (10, 10)
# Create a cat container without partitions
a = cat.Container()

# Create a byte array
buf = bytes(np.arange(int(np.prod(shape)), dtype=np.float32))

# Fill th cat container array with the buffer
a.frombuffer(shape, buf)

# Get slice from cat container
b = a[3, 6:10]

# Convert cat container to a numpy array
c = np.frombuffer(b.tobuffer(), dtype=np.float32).reshape(b.shape)
print(c)

# Squeeze b container
b.squeeze()

# Convert cat container to a numpy array
c = np.frombuffer(b.tobuffer(), dtype=np.float32).reshape(b.shape)
print(c)

# Set a value in cat container
a[:, 3] = bytes(np.full((10, 1), 3.14, np.float32))
a[1:6, 5:8] = bytes(np.full((5, 3), 0.156, np.float32))

# Convert cat container to a numpy array
d = np.frombuffer(a.tobuffer(), dtype=np.float32).reshape(a.shape)

print(d)

