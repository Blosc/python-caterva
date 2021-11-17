#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import caterva as cat
import pytest


@pytest.mark.parametrize("shape, chunks, blocks, itemsize, codec, clevel, use_dict, nthreads, filters",
                         [
                             ((100, 1230), (200, 100), (55, 3), 4, cat.Codec.LZ4, 4, 0, 1, [cat.Filter.SHUFFLE]),
                             ((234, 125), (90, 90), (20, 10), 8, cat.Codec.LZ4HC, 8, 0, 2,
                              [cat.Filter.DELTA, cat.Filter.BITSHUFFLE]),
                             ((400, 399, 401), (20, 10, 130), (6, 6, 26), 3, cat.Codec.BLOSCLZ, 5, 1, 2,
                              [cat.Filter.DELTA, cat.Filter.TRUNC_PREC])
                         ])
def test_empty(shape, chunks, blocks, itemsize, codec, clevel, use_dict, nthreads,
               filters):
    a = cat.empty(shape, chunks=chunks,
                  blocks=blocks,
                  itemsize=itemsize,
                  codec=codec,
                  clevel=clevel,
                  use_dict=use_dict,
                  nthreads=nthreads,
                  filters=filters)
    if chunks is not None:
        assert a.chunks == chunks
        assert a.blocks == blocks
    assert a.shape == shape
    assert a.itemsize == itemsize
    assert a.codec == (codec if chunks is not None else None)
    assert a.clevel == (clevel if chunks is not None else 1)
    if chunks is not None:
        assert a.filters[-len(filters):] == filters
    else:
        assert a.filters is None
