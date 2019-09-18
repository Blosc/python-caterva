import cat4py as cat
import pytest
import numpy as np
import os


@pytest.mark.parametrize("shape, pshape, itemsize, memframe, filename",
                         [
                             ([200], [20], 8, True, None),
                             ([20, 134, 13], [3, 13, 5], 4, False, None),
                             ([20, 134, 13], [3, 13, 5], 4, False, 'test_frame.cat'),
                             ([12, 13, 14, 15, 16], [2, 3, 4, 4, 4], 8, True, None)
                         ])
def test_frame(shape, pshape, itemsize, memframe, filename):
    if filename is not None and os.path.exists(filename):
        os.remove(filename)

    size = int(np.prod(shape))
    buffer = bytes(size * itemsize)
    a = cat.from_buffer(buffer, shape, pshape=pshape, itemsize=itemsize,
                        memframe=memframe, filename=filename)
    buffer1 = a.to_frame()
    buffer2 = a.to_buffer()
    # Size of a compressed frame should be less than the plain buffer for
    # the cases here
    # print("->", len(buffer1), len(buffer2))
    assert len(buffer1) < len(buffer2)

    if filename is not None and os.path.exists(filename):
        os.remove(filename)
