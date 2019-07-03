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

    a = cat.Container(pshape=pshape, itemsize=itemsize)

    size = int(np.prod(shape))

    buffer = np.arange(size, dtype=dtype).reshape(shape)

    a.from_numpy(buffer)

    buffer = buffer[slices]

    a_slice = a[slices]

    buffer2 = a_slice.to_numpy(dtype=dtype)

    np.testing.assert_almost_equal(buffer, buffer2)
