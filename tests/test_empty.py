import numpy as np
import cat4py as cat
import pytest


@pytest.mark.parametrize("shape, chunkshape, blockshape, itemsize, cname, clevel, use_dict, nthreads, filters",
                         [
                             ((100, 1230), (200, 100), (55, 3), 4, "lz4", 4, 0, 1, [1]),
                             ((23, 34), None, None, 8, "lz4hc", 8, 0, 2, [2, 1]),
                             ((400, 399, 401), (20, 10, 130), (6, 6, 26), 3, "blosclz", 5, 1, 2,
                              [4, 3, 0, 2, 1])
                         ])
def test_empty(shape, chunkshape, blockshape, itemsize, cname, clevel, use_dict, nthreads,
               filters):
    a = cat.empty(shape, chunkshape=chunkshape,
                  blockshape=blockshape,
                  itemsize=itemsize,
                  cname=cname,
                  clevel=clevel,
                  use_dict=use_dict,
                  nthreads=nthreads,
                  filters=filters)

    if chunkshape is not None:
        assert a.chunkshape == chunkshape
        assert a.blockshape == blockshape
    assert a.shape == shape
    assert a.itemsize == itemsize
    assert a.cname == (cname if chunkshape is not None else None)
    assert a.clevel == (clevel if chunkshape is not None else 1)
    if chunkshape is not None:
        assert a.filters[-len(filters):] == filters
    else:
        assert a.filters is None


@pytest.mark.parametrize("shape, chunkshape, blockshape, dtype, cname, clevel, use_dict, nthreads, filters",
                         [
                             ((100, 1230), (200, 100), (55, 3), np.float32, "lz4", 4, 0, 1, [1]),
                             ((23, 34), None, None, np.int64, "lz4hc", 8, 0, 2, [2, 1]),
                             ((400, 399, 401), (20, 10, 130), (6, 6, 26), np.int8, "blosclz",
                              5, 1, 2, [4, 3, 0, 2, 1])
                         ])
def test_empty_numpy(shape, chunkshape, blockshape, dtype, cname, clevel, use_dict, nthreads,
                     filters):
    dtype = np.dtype(dtype)
    a = cat.empty(shape, dtype.itemsize,
                  dtype=str(dtype),
                  chunkshape=chunkshape,
                  blockshape=blockshape,
                  cname=cname,
                  clevel=clevel,
                  use_dict=use_dict,
                  nthreads=nthreads,
                  filters=filters)

    if chunkshape is not None:
        assert a.chunkshape == chunkshape
        assert a.blockshape == blockshape
    assert a.shape == shape
    assert a._dtype == dtype
    assert a.itemsize == dtype.itemsize
    assert a.cname == (cname if chunkshape is not None else None)
    assert a.clevel == (clevel if chunkshape is not None else 1)
    if chunkshape is not None:
        assert a.filters[-len(filters):] == filters
    else:
        assert a.filters is None
