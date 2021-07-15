#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import caterva as cat
import numpy as np

np.random.seed(123)


shape, chunks, blocks, itemsize, codec, clevel, use_dict, nthreads, filters = (
    (400, 399, 401),
    (20, 10, 130),
    (6, 6, 26),
    3,
    cat.Codec.BLOSCLZ,
    5,
    False,
    2,
    [cat.Filter.DELTA, cat.Filter.TRUNC_PREC]
)

a = cat.empty(shape, chunks=chunks,
              blocks=blocks,
              itemsize=itemsize,
              codec=codec,
              clevel=clevel,
              use_dict=use_dict,
              nthreads=nthreads,
              filters=filters)

print("HOLA")
