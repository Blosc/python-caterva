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


@pytest.mark.parametrize("shape, chunks1, blocks1, chunks2, blocks2, itemsize",
                         [
                             ([521], [212], [33], [121], [18], 8),
                             ([20, 134, 13], [10, 43, 10], [3, 13, 5], [10, 43, 10], [3, 6, 5], 4),
                             ([12, 13, 14, 15, 16], [6, 6, 6, 6, 6], [2, 2, 2, 2, 2],
                              [7, 7, 7, 7, 7], [3, 3, 5, 3, 3], 8)
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
                             ([521], [212], [33], [121], [18], "i8"),
                             ([20, 134, 13], [10, 43, 10], [3, 13, 5], [10, 43, 10], [3, 6, 5], "f4"),
                             ([12, 13, 14, 15, 16], [6, 6, 6, 6, 6], [2, 2, 2, 2, 2],
                              [7, 7, 7, 7, 7], [3, 3, 5, 3, 3], "f8")
                         ])
def test_copy_numpy(shape, chunks1, blocks1, chunks2, blocks2, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.asarray(nparray, chunks=chunks1, blocks=blocks1)
    b = a.copy(chunks=chunks2, blocks=blocks2, complevel=5, filters=[cat.Filter.BITSHUFFLE])
    if chunks2:
        b = b[...]
    nparray2 = np.asarray(b).view(dtype)
    np.testing.assert_almost_equal(nparray, nparray2)
