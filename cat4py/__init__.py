#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


from .version import __version__

from . import container_ext as ext
# Codecs
from .container_ext import BLOSCLZ, LZ4, LZ4HC, ZLIB, ZSTD, LIZARD

# Filters
from .container_ext import NOFILTER, SHUFFLE, BITSHUFFLE, DELTA, TRUNC_PREC

# Public API for container module
from .constructors import (empty, from_buffer, from_file, from_sframe, asarray, copy)

from .ndarray import NDArray
from .ndtarray import NDTArray

# Available compression library names
cnames = list(ext.cnames2codecs)
