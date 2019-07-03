import cat4py as cat
import pytest
import numpy as np
from itertools import zip_longest as lzip


@pytest.mark.parametrize("shape, pshape1, pshape2, dtype",
                         [
                             ([4], [2], [2], np.int32),
                             ([134, 121, 78], [27, 12, 44], [12, 13, 18], np.float64),
                             ([21, 21, 21, 31], None, None, np.float32)
                         ])
def test_iters(shape, pshape1, pshape2, dtype):

    blockshape = pshape2 if pshape2 is not None else shape

    itemsize = np.dtype(dtype).itemsize

    a = cat.Container(pshape=pshape1, itemsize=itemsize)

    size = int(np.prod(shape))

    buffer = np.arange(size, dtype=dtype).reshape(shape)

    a.from_numpy(buffer)

    b = cat.Container(pshape2, itemsize=itemsize)

    for (block_r, info_r), (block_w, info_w) in lzip(a.iter_read(blockshape, dtype), b.iter_write(shape, dtype)):
        block_w[:] = block_r
