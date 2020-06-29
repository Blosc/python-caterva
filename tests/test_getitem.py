import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, chunkshape, blockshape, slices, dtype",
                         [
                             ([456], [258], [73], slice(0, 1), np.int32),
                             ([77, 134, 13], [31, 13, 5], [7, 8, 3],
                              (slice(3, 7), slice(50, 100),slice(2, 7)), np.float64),
                             ([12, 13, 14, 15, 16], None, None,
                              (slice(1, 3), slice(2, 5), slice(0, 12), slice(3, 6)), np.float32)
                         ])
def test_getitem(shape, chunkshape, blockshape, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.from_buffer(bytes(nparray), nparray.shape, itemsize=nparray.itemsize,
                        chunkshape=chunkshape, blockshape=blockshape)
    nparray_slice = nparray[slices]
    buffer_slice = np.asarray(a[slices])
    a_slice = np.frombuffer(buffer_slice, dtype=dtype).reshape(nparray_slice.shape)
    np.testing.assert_almost_equal(a_slice, nparray_slice)


@pytest.mark.parametrize("shape, chunkshape, blockshape, slices, dtype",
                         [
                             ([456], [258], [73], slice(0, 1), np.int32),
                             ([77, 134, 13], [31, 13, 5], [7, 8, 3],
                              (slice(3, 7), slice(50, 100), slice(2, 7)), np.float64),
                             ([12, 13, 14, 15, 16], None, None,
                              (slice(1, 3), slice(2, 5), slice(0, 12), slice(3, 6)), np.float32)
                         ])
def test_getitem_numpy(shape, chunkshape, blockshape, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.asarray(nparray, chunkshape=chunkshape, blockshape=blockshape)
    nparray_slice = nparray[slices]
    a_slice = a[slices]
    np.testing.assert_almost_equal(a_slice, nparray_slice)
