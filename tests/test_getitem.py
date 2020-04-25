import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, chunkshape, slices, dtype",
                         [
                             ([2], [2], slice(0, 1), np.int32),
                             ([20, 134, 13], [3, 13, 5], (slice(3, 7), slice (50, 100), slice(2, 7)), np.float64),
                             ([12, 13, 14, 15, 16], None, (slice(1, 3), slice(2, 5), slice(0, 12), slice(3, 6)), np.float32)
                         ])
def test_getitem(shape, chunkshape, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.from_buffer(bytes(nparray), nparray.shape, itemsize=nparray.itemsize, chunkshape=chunkshape)
    nparray_slice = nparray[slices]
    buffer_slice = a[slices]
    a_slice = np.frombuffer(buffer_slice, dtype=dtype).reshape(nparray_slice.shape)
    np.testing.assert_almost_equal(a_slice, nparray_slice)


@pytest.mark.parametrize("shape, chunkshape, slices, dtype",
                         [
                             ([10], [4], (slice(0, 2),), np.int32),
                             ([20, 134, 13], [3, 13, 5], (slice(3, 7), slice(50, 100), slice(2, 7)), np.float64),
                             ([12, 13, 14, 15, 16], None, (slice(1, 3), slice(2, 5), slice(0, 12), slice(3, 6)), np.float32)
                         ])
def test_getitem_numpy(shape, chunkshape, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.from_array(nparray, chunkshape=chunkshape)
    nparray_slice = nparray[slices]
    a_slice = a[slices]
    np.testing.assert_almost_equal(a_slice, nparray_slice)
