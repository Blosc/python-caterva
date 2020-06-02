import cat4py as cat
import numpy as np


shape = (100, 100)
chunkshape = (25, 25)
blockshape = (15, 15)

dtype = np.int32
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray, chunkshape=chunkshape, blockshape=blockshape)

# Add some usermeta info
usermeta = {b"author": b"cat4py example",
            b"description": b"lorem ipsum"}
assert(a.update_usermeta(usermeta) >= 0)
assert(a.get_usermeta() == usermeta)

usermeta.update({b"author": b"usermeta example"})
assert(a.update_usermeta(usermeta) >= 0)
assert(a.get_usermeta() == usermeta)
