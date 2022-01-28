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

shape = (8, 8)
chunks = (4, 4)
blocks = (2, 2)

fill_value = b"1"
a = cat.full(shape, fill_value=fill_value, chunks=chunks, blocks=blocks)

a.resize((10, 10))

print(a[:])
