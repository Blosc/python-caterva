import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, pshape, slices, dtype",
                         [
                             ([2], [2], slice(0, 1), np.int32),
                             ([20, 134, 13], [3, 13, 5], (slice(3, 7), slice (50, 100), slice(2, 7)), np.float64),
                             ([12, 13, 14, 15, 16], None, (slice(1, 3), slice(2, 5), slice(0, 12), slice(3, 6), slice(2, 7)), np.float32)
                         ])
def test_getitem(shape, pshape, slices, dtype):

    itemsize = np.dtype(dtype).itemsize

    arr = cat.Container(pshape=pshape, itemsize=itemsize)

    size = int(np.prod(shape))

    arr_np = np.arange(size, dtype=dtype).reshape(shape)

    arr.from_numpy(arr_np)

    np_sl = arr_np[slices]

    buf_sl = arr[slices]

    arr_sl = np.frombuffer(buf_sl, dtype=dtype).reshape(np_sl.shape)

    np.testing.assert_almost_equal(arr_sl, np_sl)
