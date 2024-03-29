#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import caterva as cat
import numpy as np
import os
import shutil


shape = (128, 128)
chunks = (32, 32)
blocks = (8, 8)

urlpath = "ex_persistency.cat"

if os.path.exists(urlpath):
    cat.remove(urlpath)

dtype = np.dtype(np.complex128)
itemsize = dtype.itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array (on disk)
a = cat.from_buffer(bytes(nparray), nparray.shape, itemsize, chunks=chunks, blocks=blocks,
                    urlpath=urlpath, contiguous=False)

# Read a caterva array from disk
b = cat.open(urlpath)

# Convert a caterva array to a numpy array
nparray2 = np.asarray(cat.from_buffer(b.to_buffer(), b.shape, b.itemsize)).view(dtype)

np.testing.assert_almost_equal(nparray, nparray2)

# Remove file on disk
if os.path.exists(urlpath):
    cat.remove(urlpath)
