#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


from .version import __version__

from . import caterva_ext as ext

# Public API for container module
from .constructors import (empty, zeros, full, from_buffer, open, asarray, copy)

from .ndarray import NDArray

from .utils import Codec, Filter, remove
