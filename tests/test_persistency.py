import cat4py as cat
import pytest
import numpy as np
import os


@pytest.mark.parametrize("shape, pshape, filename, dtype, copy",
                         [
                             ([2], [2], "test00.cat", np.float64, True),
                             ([20, 134, 13], [3, 13, 5], "test01.cat", np.int32, False),
                             ([12, 13, 14, 15, 16], [2, 6, 4, 5, 4], "test02.cat", np.float32, True)
                         ])
def test_persistency(shape, pshape, filename, dtype, copy):
    if os.path.exists(filename):
        os.remove(filename)

    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    cat.from_numpy(nparray, pshape=pshape, filename=filename)
    b = cat.from_file(filename, copy)
    nparray2 = b.to_numpy()
    np.testing.assert_almost_equal(nparray, nparray2)

    os.remove(filename)
