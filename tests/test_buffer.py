import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, chunkshape, blockshape, itemsize",
                         [
                             ([450], [128], [25], 8),
                             ([20, 134, 13], [3, 13, 5], [3, 10, 5], 4),
                             ([12, 13, 14, 15, 16], None, None, 8)
                         ])
def test_buffer(shape, chunkshape, blockshape, itemsize):
    size = int(np.prod(shape))
    buffer = bytes(size * itemsize)
    a = cat.from_buffer(buffer, shape, itemsize, chunkshape=chunkshape, blockshape=blockshape)
    buffer2 = a.to_buffer()
    assert buffer == buffer2
