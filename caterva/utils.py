#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from enum import Enum
import os
import shutil


class Codec(Enum):
    """
    Available codecs.
    """
    BLOSCLZ = 0
    LZ4 = 1
    LZ4HC = 2
    ZLIB = 4
    ZSTD = 5


class Filter(Enum):
    """
    Available filters.
    """
    NOFILTER = 0
    SHUFFLE = 1
    BITSHUFFLE = 2
    DELTA = 3
    TRUNC_PREC = 4


def remove(urlpath):
    """
    Remove a caterva file.

    Parameters
    ----------
    urlpath: String
        The array urlpath.
    """
    if os.path.exists(urlpath):
        if os.path.isdir(urlpath):
            shutil.rmtree(urlpath)
        else:
            os.remove(urlpath)
