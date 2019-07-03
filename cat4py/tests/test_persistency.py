import cat4py as cat
import pytest
import numpy as np
import os


@pytest.mark.parametrize("shape, pshape, filename, dtype",
                         [
                             ([2], [2], "test00.cat", np.float64),
                             ([20, 134, 13], [3, 13, 5], "test01.cat", np.int32),
                             ([12, 13, 14, 15, 16], [2, 6, 4, 5, 4], "test02.cat", np.float32)
                         ])
def test_persistency(shape, pshape, filename, dtype):

    itemsize = np.dtype(dtype).itemsize

    a = cat.Container(pshape=pshape, filename=filename, itemsize=itemsize)

    size = int(np.prod(shape))

    buffer = np.arange(size, dtype=dtype).reshape(shape)

    a.from_numpy(buffer)

    b = cat.from_file(filename)

    buffer2 = b.to_numpy(dtype)

    np.testing.assert_almost_equal(buffer, buffer2)

    os.remove(filename)
