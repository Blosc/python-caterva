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
from msgpack import packb


@pytest.mark.parametrize("contiguous",
                         [
                             True,
                             False,
                         ])
@pytest.mark.parametrize("shape, chunks, blocks, urlpath, dtype",
                         [
                             ([556], [221], [33], "testmeta00.cat", np.float64),
                             ([20, 134, 13], [12, 66, 8], [3, 13, 5], "testmeta01.cat", np.int32),
                             ([12, 13, 14, 15, 16], [8, 9, 4, 12, 9], [2, 6, 4, 5, 4], "testmeta02.cat", np.float32)
                         ])
def test_metalayers(shape, chunks, blocks, urlpath, contiguous, dtype):
    if os.path.exists(urlpath):
        cat.remove(urlpath)

    numpy_meta = packb({b"dtype": str(np.dtype(dtype))})
    test_meta = packb({b"lorem": 1234})

    # Create an empty caterva array (on disk)
    itemsize = np.dtype(dtype).itemsize
    a = cat.empty(shape, itemsize, chunks=chunks, blocks=blocks,
                  urlpath=urlpath, contiguous=contiguous,
                  meta={"numpy": numpy_meta,
                        "test": test_meta})

    assert ("numpy" in a.meta)
    assert ("error" not in a.meta)
    assert (a.meta["numpy"] == numpy_meta)
    assert ("test" in a.meta)
    assert (a.meta["test"] == test_meta)

    test_meta = packb({b"lorem": 4231})
    a.meta["test"] = test_meta
    assert (a.meta["test"] == test_meta)

    # Remove file on disk
    cat.remove(urlpath)
