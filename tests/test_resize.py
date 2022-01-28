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


@pytest.mark.parametrize("shape, new_shape, chunks, blocks, fill_value",
                         [
                             ((100, 1230), (200, 1230), (200, 100), (55, 3), b"0123"),
                             ((23, 34), (23, 120), (20, 20), (10, 10), b"sun"),
                             ((80, 51, 60), (80, 100, 100), (20, 10, 33), (6, 6, 26), b"qwerty")
                         ])
def test_resize(shape, new_shape, chunks, blocks, fill_value):
    a = cat.full(shape, fill_value=fill_value, chunks=chunks, blocks=blocks)

    a.resize(new_shape)
    assert a.shape == new_shape

    slices = tuple(slice(s) for s in shape)
    for i in np.nditer(np.array(a[slices])):
        assert i == fill_value
