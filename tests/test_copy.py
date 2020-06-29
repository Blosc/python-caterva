import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, chunkshape1, blockshape1, chunkshape2, blockshape2, itemsize",
                         [
                             ([521], [212], [33], [121], [18], 8),
                             ([20, 134, 13], [10, 43, 10], [3, 13, 5], None, None, 4),
                             ([12, 13, 14, 15, 16], None, None, [7, 7, 7, 7, 7], [3, 3, 5, 3, 3], 8)
                         ])
def test_copy(shape, chunkshape1, blockshape1, chunkshape2, blockshape2, itemsize):
    size = int(np.prod(shape))
    buffer = bytes(size * itemsize)
    a = cat.from_buffer(buffer, shape, itemsize, chunkshape=chunkshape1, blockshape=blockshape1,
                        complevel=2)
    b = a.copy(chunkshape=chunkshape2, blockshape=blockshape2,
               itemsize=itemsize, complevel=5, filters=[2])
    buffer2 = b.to_buffer()
    assert buffer == buffer2


@pytest.mark.parametrize("shape, chunkshape1, blockshape1, chunkshape2, blockshape2, dtype",
                         [
                             ([521], [212], [33], [121], [18], np.float64),
                             ([20, 134, 13], [10, 43, 10], [3, 13, 5], None, None, np.int32),
                             ([12, 13, 14, 15, 16], None, None, [7, 7, 7, 7, 7], [3, 3, 5, 3, 3],
                              np.float32)
                         ])
def test_copy_numpy(shape, chunkshape1, blockshape1, chunkshape2, blockshape2, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.asarray(nparray, chunkshape1=chunkshape1, blockshape=blockshape1)
    b = a.copy(chunkshape=chunkshape2, blockshape=blockshape2, complevel=5, filters=[2])
    if chunkshape2 is not None:
        nparray2 = np.asarray(b.copy())
    else:
        nparray2 = b
    np.testing.assert_almost_equal(nparray, nparray2)
