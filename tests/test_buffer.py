import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, pshape, itemsize",
                         [
                             ([2], [2], 8),
                             ([20, 134, 13], [3, 13, 5], 4),
                             ([12, 13, 14, 15, 16], None, 8)
                         ])
def test_buffer(shape, pshape, itemsize):
    size = int(np.prod(shape))
    buffer = bytes(size * itemsize)
    a = cat.from_buffer(buffer, shape, pshape=pshape, itemsize=itemsize)
    buffer2 = a.to_buffer()
    assert buffer == buffer2
