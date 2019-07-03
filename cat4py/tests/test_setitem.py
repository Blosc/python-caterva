import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, pshape, slices, dtype",
                         [
                             ([2], None, slice(0, 1), np.int32),
                             ([20, 134, 13], None, (slice(3, 7), slice (50, 100), slice(2, 7)), np.float64),
                             ([12, 13, 14, 15, 16], None, (slice(1, 3), slice(2, 5), slice(0, 12), slice(3, 6), slice(2, 7)), np.float32)
                         ])
def test_setitem(shape, pshape, slices, dtype):

    itemsize = np.dtype(dtype).itemsize

    a = cat.Container(pshape=pshape, itemsize=itemsize)

    size = int(np.prod(shape))

    buffer = np.arange(size, dtype=dtype).reshape(shape)

    a.from_numpy(buffer)

    slice_shape = slices.stop - slices.start if isinstance(slices, slice) else [s.stop - s.start for s in slices]

    buffer[slices] = np.ones(slice_shape, dtype=dtype)

    a[slices] = np.ones(slice_shape, dtype=dtype)

    buffer2 = a.to_numpy(dtype=dtype)

    np.testing.assert_almost_equal(buffer, buffer2)
