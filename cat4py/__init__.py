#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from .container_ext import CParams, DParams, Container  # the order of the import is important: extensions first
from .version import version as __version__
