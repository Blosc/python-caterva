import cat4py as cat
import numpy as np
from time import time


shape = (100, 500, 250)
chunkshape = (20, 100, 50)
blockshape = (10, 50, 25)

enforceframe = True

if (enforceframe):
    print("Testing a frame...")
else:
    print("Testing a schunk...")

dtype = np.float64

# Compression properties
cname = "zstd"
clevel = 6
filter = cat.SHUFFLE
nthreads = 2


content = np.linspace(0, 10, int(np.prod(shape)), dtype=dtype).reshape(shape)

t0 = time()
a = cat.empty(shape, chunkshape=chunkshape, blockshape=blockshape,
              itemsize=content.itemsize, enforceframe=enforceframe,
              cname=cname, clevel=clevel, filters=[filter], nthreads=nthreads)
for block, info in a.iter_write():
    nparray = content[info.slice]
    block[:] = bytes(nparray)
t1 = time()
print("Time for filling array (caterva): %.3fs ; CRatio: %.1fx" % ((t1 - t0), a.compratio))

# Setup the coordinates for random planes
planes_idx = np.random.randint(0, shape[1], 50)

# Time getitem with caterva
t0 = time()
for i in planes_idx:
    rbytes = a[:, i, :]
    block = np.frombuffer(rbytes, dtype=dtype).reshape((shape[0], shape[2]))
t1 = time()
print("Time for reading with getitem (caterva): %.3fs" % (t1 - t0))

