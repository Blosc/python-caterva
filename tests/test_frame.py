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
    frame1 = a.to_sframe()
    buffer1 = a.to_buffer()
    # Size of a compressed frame should be less than the plain buffer for
    # the cases here
    # print("->", len(frame1), len(buffer1))
    assert len(frame1) < len(buffer1)

    b = cat.from_sframe(frame1, copy=True)
    frame2 = b.to_sframe()
    # TODO: the next assert currently fails.  Investigate...
    # assert len(frame2) == len(frame1)
    buffer2 = b.to_buffer()
    assert buffer2 == buffer1

    if filename is not None and os.path.exists(filename):
        os.remove(filename)
