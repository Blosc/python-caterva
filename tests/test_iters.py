import cat4py as cat
import pytest
import numpy as np
from itertools import zip_longest as lzip


@pytest.mark.parametrize("shape, chunkshape1, chunkshape2, dtype",
                         [
                             ([4], [2], [2], np.int32),
                             ([134, 121, 78], [27, 12, 44], [12, 13, 18], np.float64),
                             ([21, 21, 21, 31], None, None, np.float32),
                             ([21, 10, 50, 15], None, [13, 5, 13, 2], np.dtype("S3"))
                         ])
def test_iters(shape, chunkshape1, chunkshape2, dtype):
    size = int(np.prod(shape))
    nparray = np.ones(size, dtype=dtype).reshape(shape)
    a = cat.from_buffer(bytes(nparray), nparray.shape, itemsize=nparray.itemsize, chunkshape=chunkshape1)

    itemsize = np.dtype(dtype).itemsize
    b = cat.empty(shape, chunkshape=chunkshape2, itemsize=itemsize)
    itershape = chunkshape2 if chunkshape2 is not None else b.shape

    for (block_r, info_r), (block_w, info_w) in lzip(a.iter_read(itershape), b.iter_write()):
        block_w[:] = block_r

    nparray2 = b.to_numpy(dtype)
    np.testing.assert_equal(nparray, nparray2)


@pytest.mark.parametrize("shape, chunkshape1, chunkshape2, dtype",
                         [
                             ([4], [2], [2], np.int32),
                             ([134, 121, 78], [27, 12, 44], [12, 13, 18], np.float64),
                             ([21, 21, 21, 31], None, None, np.float32),
                             ([21, 10, 50, 15], None, [13, 5, 13, 2], np.dtype("S3"))
                         ])
def test_iters_numpy(shape, chunkshape1, chunkshape2, dtype):
    size = int(np.prod(shape))
    nparray = np.ones(size, dtype=dtype).reshape(shape)
    a = cat.from_numpy(nparray, chunkshape=chunkshape1)  # creates a NPArray

    b = cat.empty(shape, dtype, chunkshape=chunkshape2)
    itershape = chunkshape2 if chunkshape2 is not None else b.shape

    for (block_r, info_r), (block_w, info_w) in lzip(a.iter_read(itershape), b.iter_write()):
        block_w[:] = block_r

    nparray2 = b.to_numpy()
    np.testing.assert_equal(nparray, nparray2)
