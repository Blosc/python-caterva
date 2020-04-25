import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, chunkshape, itemsize",
                         [
                             ([24], [3], 8),
                             ([20, 134, 13], [3, 13, 5], 4),
                             ([12, 13, 14, 15, 16], None, 8)
                         ])
def test_buffer(shape, chunkshape, itemsize):
    size = int(np.prod(shape))
    buffer = bytes(size * itemsize)
    a = cat.from_buffer(buffer, shape, chunkshape=chunkshape, itemsize=itemsize)
    buffer2 = a.to_buffer()
    assert buffer == buffer2
