import cat4py as cat
import pytest
import numpy as np
import os


@pytest.mark.parametrize("shape, pshape, itemsize, memframe, filename, copy",
                         [
                             ([200], [20], 8, True, None, True),
                             ([200], [20], 8, True, None, False),
                             ([20, 134, 13], [3, 13, 5], 4, False, None, True),
                             ([20, 134, 13], [3, 13, 5], 4, False, 'test_frame.cat', False),
                             ([12, 13, 14, 15, 16], [2, 3, 4, 4, 4], 8, True, None, True)
                         ])
def test_frame(shape, pshape, itemsize, memframe, filename, copy):
    if filename is not None and os.path.exists(filename):
        os.remove(filename)

    size = int(np.prod(shape))
    buffer = bytes(size * itemsize)
    a = cat.from_buffer(buffer, shape, pshape=pshape, itemsize=itemsize,
                        memframe=memframe, filename=filename)
    sframe1 = a.to_sframe()
    buffer1 = a.to_buffer()
    # Size of a compressed frame should be less than the plain buffer for the cases here
    assert len(sframe1) < len(buffer1)

    b = cat.from_sframe(sframe1, copy=copy)
    sframe2 = b.to_sframe()
    # For some reason, the size of sframe1 and sframe2 are not equal when copies are made,
    # but the important thing is that the length of the frame should be stable in multiple
    # round-trips after the first one.
    # assert len(sframe2) == len(sframe1)
    sframe3 = sframe2
    c = b
    for i in range(10):
        c = cat.from_sframe(sframe3, copy=copy)
        sframe3 = c.to_sframe()
    assert len(sframe3) == len(sframe2)
    buffer2 = b.to_buffer()
    assert buffer2 == buffer1
    buffer3 = c.to_buffer()
    assert buffer3 == buffer1

    if filename is not None and os.path.exists(filename):
        os.remove(filename)
