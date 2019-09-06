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
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.from_numpy(nparray, pshape=pshape, itemsize=nparray.itemsize)

    nparray_slice = nparray[slices]
    buffer_slice = a[slices]
    a_slice = np.frombuffer(buffer_slice, dtype=dtype).reshape(nparray_slice.shape)
    np.testing.assert_almost_equal(a_slice, nparray_slice)
