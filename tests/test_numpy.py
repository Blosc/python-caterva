import cat4py as cat
import pytest
import numpy as np


@pytest.mark.parametrize("shape, chunkshape, blockshape, dtype",
                         [
                             ([931], [223], [45], np.int32),
                             ([134, 121, 78], [12, 13, 18], [4, 4, 9], np.float64),
                             ([21, 21, 21, 31], None, None, np.float32)
                         ])
def test_numpy(shape, chunkshape, blockshape, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = cat.asarray(nparray, chunkshape=chunkshape, blockshape=blockshape)
    nparray2 = a.to_numpy()
    np.testing.assert_almost_equal(nparray, nparray2)
