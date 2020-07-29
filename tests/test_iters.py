import cat4py as cat
import pytest
import numpy as np
from itertools import zip_longest as lzip


@pytest.mark.parametrize("shape, chunkshape1, blockshape1, chunkshape2, blockshape2, dtype",
                         [
                             ([412], [231], [40], [121], [33], np.int32),
                             ([134, 121, 78], [27, 12, 44], [10, 10, 10], [43, 77, 40],
                              [12, 13, 18], np.float64),
                             ([21, 21, 21, 31], None, None, None, None, np.float32),
                             ([21, 10, 50, 15], None, None, [13, 5, 13, 2], [13, 5, 13, 2],
                              "S3")
                         ])
def test_iters(shape, chunkshape1, blockshape1, chunkshape2, blockshape2, dtype):
    dtype = np.dtype(dtype)
    size = int(np.prod(shape))
    nparray = np.ones(size, dtype=dtype).reshape(shape)
    a = cat.asarray(nparray, chunkshape=chunkshape1, blockshape=blockshape1)

    itemsize = dtype.itemsize
    b = cat.empty(shape, itemsize, chunkshape=chunkshape2, blockshape=blockshape2)
    itershape = chunkshape2 if chunkshape2 is not None else b.shape

    for (block_r, info_r), (block_w, info_w) in lzip(a.iter_read(itershape), b.iter_write()):
        block_w[:] = bytearray(np.asarray(block_r))

    nparray2 = np.asarray(cat.from_buffer(b.to_buffer(), b.shape, b.itemsize, str(dtype)))
    np.testing.assert_equal(nparray, nparray2)


@pytest.mark.parametrize("shape, chunkshape1, blockshape1, chunkshape2, blockshape2, dtype",
                         [
                             ([412], [231], [40], [121], [33], np.int32),
                             ([134, 121, 78], [27, 12, 44], [10, 10, 10], [43, 77, 40],
                              [12, 13, 18], np.float64),
                             ([21, 21, 21, 31], None, None, None, None, np.float32),
                             ([21, 10, 50, 15], None, None, [13, 5, 13, 2], [13, 5, 13, 2],
                              "S3")
                         ])
def test_iters_numpy(shape, chunkshape1, blockshape1, chunkshape2, blockshape2, dtype):
    dtype = np.dtype(dtype)

    size = int(np.prod(shape))
    nparray = np.ones(size, dtype=dtype).reshape(shape)
    a = cat.asarray(nparray, chunkshape=chunkshape1, blockshape=blockshape1)  # creates a NPArray

    b = cat.empty(shape, dtype.itemsize, str(dtype), chunkshape=chunkshape2, blockshape=blockshape2)
    itershape = chunkshape2 if chunkshape2 is not None else b.shape

    for (block_r, info_r), (block_w, info_w) in lzip(a.iter_read(itershape), b.iter_write()):
        block_w[:] = bytearray(np.asarray(block_r))

    nparray2 = np.asarray(b.copy())
    np.testing.assert_equal(nparray, nparray2)
