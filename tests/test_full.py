#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import caterva as cat
import pytest


@pytest.mark.parametrize("shape, chunks, blocks, fill_value, cname, clevel, use_dict, nthreads",
                         [
                             ((100, 1230), (200, 100), (55, 3), b"0123", cat.Codec.LZ4HC, 4, 0, 1),
                             ((23, 34), (20, 20), (10, 10), b"sun", cat.Codec.LZ4HC, 8, 0, 2),
                             ((80, 51, 60), (20, 10, 33), (6, 6, 26), b"qwerty", cat.Codec.ZLIB, 5, 1, 2)
                         ])
def test_full(shape, chunks, blocks, fill_value, cname, clevel, use_dict, nthreads):
    a = cat.full(shape, fill_value=fill_value, chunks=chunks, blocks=blocks, cname=cname, clevel=clevel,
                 use_dict=use_dict, nthreads=nthreads)

    for i in np.nditer(np.array(a[:])):
        assert i == fill_value
