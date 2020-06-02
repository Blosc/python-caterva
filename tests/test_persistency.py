import cat4py as cat
import pytest
import numpy as np
import os


@pytest.mark.parametrize("shape, chunkshape, blockshape, filename, dtype, copy",
                         [
                             ([634], [156], [33], "test00.cat", np.float64, True),
                             ([20, 134, 13], [7, 22, 5], [3, 5, 3], "test01.cat", np.int32, False),
                             ([12, 13, 14, 15, 16], [4, 6, 4, 7, 5], [2, 4, 2, 3, 3], "test02.cat", np.float32, True)
                         ])
def test_persistency(shape, chunkshape, blockshape, filename, dtype, copy):
    if os.path.exists(filename):
        os.remove(filename)

    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    cat.from_numpy(nparray, chunkshape=chunkshape, blockshape=blockshape, enforceframe=True, filename=filename)
    b = cat.from_file(filename, copy)
    nparray2 = b.to_numpy()
    np.testing.assert_almost_equal(nparray, nparray2)

    os.remove(filename)
