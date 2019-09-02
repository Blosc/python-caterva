import cat4py as cat
import numpy as np
import os

pshape = (5, 5)
shape = (10, 10)
filename = "ex_usermeta.cat"
if (os.path.exists(filename)):
    # Remove file on disk
    os.remove(filename)

dtype = np.int32
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = cat.from_numpy(nparray, pshape, itemsize=itemsize, filename=filename)

# Add some usermeta info
usermeta = {b"author": b"cat4py example",
            b"description": b"lorem ipsum"}
assert(a.update_usermeta(usermeta) >= 0)
assert(a.get_usermeta() == usermeta)

usermeta.update({b"author": b"usermeta example"})
assert(a.update_usermeta(usermeta) >= 0)
assert(a.get_usermeta() == usermeta)

print("File is available at:", os.path.abspath(filename))
