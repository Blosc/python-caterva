import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([2], [2], np.int32),
                             ([134, 121, 78], [12, 13, 18], np.float64),
                             ([21, 21, 21, 31], None, np.float32)
                         ])
def test_numpy(shape, pshape, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.from_numpy(nparray, pshape=pshape)
    nparray2 = a.to_numpy()
    np.testing.assert_almost_equal(nparray, nparray2)
