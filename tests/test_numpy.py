import caterva as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, chunks, blocks, dtype",
                         [
                             ([931], [223], [45], np.int32),
                             ([134, 121, 78], [12, 13, 18], [4, 4, 9], np.float64),
                             ([21, 21, 21, 31], None, None, np.float32)
                         ])
def test_numpy(shape, chunks, blocks, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.asarray(nparray, chunks=chunks, blocks=blocks)
    nparray2 = np.asarray(a.copy()).view(dtype)
    np.testing.assert_almost_equal(nparray, nparray2)
