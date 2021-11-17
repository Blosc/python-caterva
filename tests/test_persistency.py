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
import os


@pytest.mark.parametrize("sequencial",
                         [
                             True,
                             False,
                         ])
@pytest.mark.parametrize("shape, chunks, blocks, urlpath, dtype",
                         [
                             ([634], [156], [33], "test00.cat", np.float64),
                             ([20, 134, 13], [7, 22, 5], [3, 5, 3], "test01.cat", np.int32),
                             ([12, 13, 14, 15, 16], [4, 6, 4, 7, 5], [2, 4, 2, 3, 3], "test02.cat", np.float32)
                         ])
def test_persistency(shape, chunks, blocks, urlpath, sequencial, dtype):
    if os.path.exists(urlpath):
        cat.remove(urlpath)

    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    _ = cat.asarray(nparray, chunks=chunks, blocks=blocks,
                    urlpath=urlpath, sequencial=sequencial)
    b = cat.open(urlpath)

    bc = b[:]

    nparray2 = np.asarray(bc).view(dtype)
    np.testing.assert_almost_equal(nparray, nparray2)

    cat.remove(urlpath)
