import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, pshape1, pshape2, itemsize",
                         [
                             ([2], [2], [2], 8),
                             ([20, 134, 13], [3, 13, 5], [3, 2, 4], 4),
                             ([12, 13, 14, 15, 16], None, [3, 3, 5, 3, 3], 8)
                         ])
def ptest_copy(shape, pshape1, pshape2, itemsize):

    size = int(np.prod(shape))

    buffer = bytes(size * itemsize)

    a = cat.from_buffer(buffer, shape, pshape1, itemsize=itemsize, clevel=2)

    b = a.copy(pshape=pshape2, itemsize=itemsize, clevel=5, filters=[2])

    buffer2 = b.to_buffer()

    assert buffer == buffer2
