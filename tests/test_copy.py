import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, chunkshape1, chunkshape2, itemsize",
                         [
                             ([2], [2], [2], 8),
                             ([20, 134, 13], [3, 13, 5], [3, 2, 4], 4),
                             ([12, 13, 14, 15, 16], None, [3, 3, 5, 3, 3], 8)
                         ])
def test_copy(shape, chunkshape1, chunkshape2, itemsize):
    size = int(np.prod(shape))
    buffer = bytes(size * itemsize)
    a = cat.from_buffer(buffer, shape, chunkshape=chunkshape1, itemsize=itemsize, complevel=2)
    b = a.copy(chunkshape=chunkshape2, itemsize=itemsize, complevel=5, filters=[2])
    buffer2 = b.to_buffer()
    assert buffer == buffer2


@pytest.mark.parametrize("shape, chunkshape1, chunkshape2, dtype",
                         [
                             ([2], [2], [2], np.float64),
                             ([20, 134, 13], [3, 13, 5], None, np.int32),
                             ([12, 13, 14, 15, 16], None, [3, 3, 5, 3, 3], np.float32)
                         ])
def test_copy_numpy(shape, chunkshape1, chunkshape2, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.from_numpy(nparray)
    b = a.copy(chunkshape=chunkshape2, complevel=5, filters=[2])
    if chunkshape2 is not None:
        nparray2 = b.to_numpy()
    else:
        nparray2 = b
    np.testing.assert_almost_equal(nparray, nparray2)
