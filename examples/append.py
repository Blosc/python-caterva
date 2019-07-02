import cat4py as cat
import numpy as np
from itertools import zip_longest as lzip

pshape = (4, 5)
shape = (10, 12)
blockshape = pshape
dtype = np.float32

# Create a cat container
a = cat.Container(itemsize=4, clevel=9)
b = cat.Container(pshape, itemsize=4)

# Fill a
np_array = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape) * np.pi / 2
a.from_numpy(np_array)


# Fill b from a
for (block_r, r_info), (block_w, w_info) in lzip(a.iter_read(blockshape, np.float32), b.iter_write(shape, np.float32)):
    block_w[:] = np.sin(block_r)

print(b.to_numpy(dtype))
