import cat4py as cat
import numpy as np

cparams = cat.CParams(itemsize=4)  # we will be dealing with itemsizes of 4's
dparams = cat.DParams()

pshape = (5, 5)
shape = (10, 10)
filename = "persistency-container.cat"


# Create a cat container with partitions
data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)

a = cat.fromnumpy(data)

c = cat.tonumpy(a, dtype=np.float32)
print(c)