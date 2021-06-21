import caterva as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, chunks1, blocks1, chunks2, blocks2, itemsize",
                         [
                             ([521], [212], [33], [121], [18], 8),
                             ([20, 134, 13], [10, 43, 10], [3, 13, 5], None, None, 4),
                             ([12, 13, 14, 15, 16], None, None, [7, 7, 7, 7, 7], [3, 3, 5, 3, 3], 8)
                         ])
def test_copy(shape, chunks1, blocks1, chunks2, blocks2, itemsize):
    size = int(np.prod(shape))
    buffer = bytes(size * itemsize)
    a = cat.from_buffer(buffer, shape, itemsize, chunks=chunks1, blocks=blocks1,
                        complevel=2)
    b = a.copy(chunks=chunks2, blocks=blocks2,
               itemsize=itemsize, complevel=5, filters=[cat.Filter.BITSHUFFLE])
    buffer2 = b.to_buffer()
    assert buffer == buffer2


@pytest.mark.parametrize("shape, chunks1, blocks1, chunks2, blocks2, dtype",
                         [
                             ([521], [212], [33], [121], [18], np.float64),
                             ([20, 134, 13], [10, 43, 10], [3, 13, 5], None, None, np.int32),
                             ([12, 13, 14, 15, 16], None, None, [7, 7, 7, 7, 7], [3, 3, 5, 3, 3],
                              np.float32)
                         ])
def test_copy_numpy(shape, chunks1, blocks1, chunks2, blocks2, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.asarray(nparray, chunks1=chunks1, blocks=blocks1)
    b = a.copy(chunks=chunks2, blocks=blocks2, complevel=5, filters=[cat.Filter.BITSHUFFLE])
    if chunks2:
        b = b.copy()
    nparray2 = np.asarray(b).view(dtype)
    np.testing.assert_almost_equal(nparray, nparray2)
