#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Codecs
BLOSCLZ = 0
LZ4 = 1
LZ4HC = 2
ZLIB = 4
LIZARD = 5

# Filters
NOFILTER = 0
SHUFFLE = 1
BITSHUFFLE = 2
DELTA = 3
TRUNC_PREC = 4

from .high_level import Container, _WriteIter, _ReadIter, empty, from_buffer, from_numpy, from_file
from .version import version as __version__
