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
def test_persistency(shape, pshape, filename, dtype):

    itemsize = np.dtype(dtype).itemsize

    # Create a numpy array
    nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

    # Create a caterva array from a numpy array
    a = cat.from_numpy(nparray, pshape, itemsize=itemsize)

    # Create an empty caterva array (on disk)
    b = cat.empty(shape, pshape, filename, itemsize=itemsize, metalayers={"numpy": {b"dtype": str(np.dtype(dtype))},
                                                                          "test": {b"lorem": 1234}})

    assert (b.has_metalayer("numpy") is True)

    assert (b.get_metalayer("numpy") == {b"dtype": bytes(str(np.dtype(dtype)), "utf-8")})

    assert (b.has_metalayer("test") is True)

    assert (b.get_metalayer("test") == {b"lorem": 1234})

    assert (b.update_metalayer("test", {b"lorem": 4321}) >= 0)

    assert (b.get_metalayer("test") == {b"lorem": 4321})

    # Fill an empty caterva array using a block iterator
    for block, info in b.iter_write():
        block[:] = bytes(nparray[info.slice])

    assert (b.update_user_metalayer({b"author": b"cat4py example", b"description": b"lorem ipsum"}) >= 0)

    assert (b.get_user_metalayer() == {b"author": b"cat4py example", b"description": b"lorem ipsum"})

    assert (b.update_user_metalayer({b"author": b"cat4py example"}) >= 0)

    assert (b.get_user_metalayer() == {b"author": b"cat4py example"})

    # Remove file on disk
    os.remove(filename)
