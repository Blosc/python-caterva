import cat4py as cat
import pytest
import numpy as np
import os


@pytest.mark.parametrize("shape, pshape, filename, dtype",
                         [
                             ([2], [2], "testmeta00.cat", np.float64),
                             ([20, 134, 13], [3, 13, 5], "testmeta01.cat", np.int32),
                             ([12, 13, 14, 15, 16], [2, 6, 4, 5, 4], "testmeta02.cat", np.float32)
                         ])
def test_metalayers(shape, pshape, filename, dtype):
    if os.path.exists(filename):
        os.remove(filename)

    # Create an empty caterva array (on disk)
    itemsize = np.dtype(dtype).itemsize
    a = cat.empty(shape, pshape=pshape, filename=filename, itemsize=itemsize,
                  metalayers={"numpy": {b"dtype": str(np.dtype(dtype))},
                              "test": {b"lorem": 1234}})

    assert (a.has_metalayer("numpy") is True)
    assert (a.get_metalayer("error") is None)
    assert (a.get_metalayer("numpy") == {b"dtype": bytes(str(np.dtype(dtype)), "utf-8")})
    assert (a.has_metalayer("test") is True)
    assert (a.get_metalayer("test") == {b"lorem": 1234})
    assert (a.update_metalayer("test", {b"lorem": 4321}) >= 0)
    assert (a.get_metalayer("test") == {b"lorem": 4321})

    # Fill an empty caterva array using a block iterator
    nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    for block, info in a.iter_write():
        block[:] = bytes(nparray[info.slice])

    assert (a.update_usermeta({b"author": b"cat4py example", b"description": b"lorem ipsum"}) >= 0)
    assert (a.get_usermeta() == {b"author": b"cat4py example", b"description": b"lorem ipsum"})
    assert (a.update_usermeta({b"author": b"cat4py example"}) >= 0)
    assert (a.get_usermeta() == {b"author": b"cat4py example"})

    # Remove file on disk
    os.remove(filename)
