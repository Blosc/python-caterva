#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import caterva as cat
import pytest
import numpy as np


argnames = "shape, chunks, blocks, slices, dtype"
argvalues = [
    ([456], [258], [73], slice(0, 1), np.int32),
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7),
     np.float64),
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 3), ..., slice(3, 6)),
     np.float32)
]


@pytest.mark.parametrize(argnames, argvalues)
def test_getitem(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.from_buffer(bytes(nparray), nparray.shape, nparray.itemsize,
                        chunks=chunks, blocks=blocks)
    nparray_slice = nparray[slices]
    buffer_slice = np.asarray(a[slices])
    a_slice = np.frombuffer(buffer_slice, dtype=dtype).reshape(nparray_slice.shape)
    np.testing.assert_almost_equal(a_slice, nparray_slice)


@pytest.mark.parametrize(argnames, argvalues)
def test_getitem_numpy(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.asarray(nparray, chunks=chunks, blocks=blocks)
    nparray_slice = nparray[slices]
    a_slice = np.asarray(a[slices]).view(dtype)

    np.testing.assert_almost_equal(a_slice, nparray_slice)
