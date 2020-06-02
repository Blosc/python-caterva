import cat4py as cat
import numpy as np
import os
from itertools import zip_longest as lzip

shape = (156, 223)
chunkshape = (22, 32)
blockshape = (12, 7)

filename = "ex_metalayers.cat"
if os.path.exists(filename):
    # Remove file on disk
    os.remove(filename)

dtype = np.int32
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray, chunkshape=chunkshape, blockshape=blockshape,
                   enforceframe=True)

# Create an empty caterva array (on disk)
metalayers = {"numpy": {b"dtype": str(np.dtype(dtype))},
              "test": {b"lorem": 1234}
              }
b = cat.empty(shape, chunkshape=chunkshape, blockshape=blockshape,
                  enforceframe=True, filename=filename, itemsize=itemsize,
                  metalayers=metalayers)

assert(b.has_metalayer("numpy") is True)
assert(b.get_metalayer("numpy") == {b"dtype": bytes(str(np.dtype(dtype)), "utf-8")})
assert(b.has_metalayer("test") is True)
assert(b.get_metalayer("test") == {b"lorem": 1234})
assert(b.update_metalayer("test", {b"lorem": 4321}) >= 0)
assert(b.get_metalayer("test") == {b"lorem": 4321})

# Fill an empty caterva array using a block iterator
for block, info in b.iter_write():
    block[:] = bytes(nparray[info.slice])

# Assert that both caterva arrays are equal
itershape = (5, 5)
for (block1, info1), (block2, info2) in lzip(a.iter_read(blockshape), b.iter_read(blockshape)):
    assert bytes(block1) == block2

assert(b.update_usermeta({b"author": b"cat4py example", b"description": b"lorem ipsum"}) >= 0)
assert(b.get_usermeta() == {b"author": b"cat4py example", b"description": b"lorem ipsum"})

assert(b.update_usermeta({b"author": b"cat4py example"}) >= 0)
assert(b.get_usermeta() == {b"author": b"cat4py example"})

# Remove file on disk
os.remove(filename)
