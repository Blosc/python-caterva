import struct
import cat4py as cat
import numpy as np


pshape = (5, 5)
shape = (10, 10)
slices = (slice(2, 4), slice(1, 4))

dtype = np.float32

itemsize = np.dtype(dtype).itemsize

arr = cat.Container(pshape=pshape, itemsize=itemsize)

size = int(np.prod(shape))

arr_np = np.arange(size, dtype=dtype).reshape(shape)

arr.from_numpy(arr_np)

np_sl = arr_np[slices]

buf_sl = arr[slices]

arr_sl = np.frombuffer(buf_sl, dtype=dtype).reshape(np_sl.shape)

np.testing.assert_almost_equal(arr_sl, np_sl)
