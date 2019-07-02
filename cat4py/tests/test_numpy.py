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

    itemsize = np.dtype(dtype).itemsize

    a = cat.Container(pshape=pshape, itemsize=itemsize)

    size = int(np.prod(shape))

    buffer = np.arange(size, dtype=dtype).reshape(shape)

    a.from_numpy(buffer)

    buffer2 = a.to_numpy(dtype)

    np.testing.assert_almost_equal(buffer, buffer2)
