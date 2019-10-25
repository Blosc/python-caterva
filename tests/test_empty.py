import numpy as np
import cat4py as cat
import pytest


@pytest.mark.parametrize("shape, pshape, itemsize, cname, clevel, use_dict, cnthreads, dnthreads, blocksize, filters",
                         [
                             ((100, 1230), (20, 10), 4, "lz4", 4, 0, 1, 1, 0, [1]),
                             ((23, 34), None, 8, "lz4hc", 8, 0, 2, 1, 256 * 1024, [2, 1]),
                             ((400, 399, 401), (20, 10, 130), 3, "blosclz", 5, 1, 2, 2, 128 * 1024, [4, 3, 0, 2, 1])
                         ])
def test_empty(shape, pshape, itemsize, cname, clevel, use_dict, cnthreads, dnthreads, blocksize, filters):
    a = cat.empty(shape, pshape=pshape,
                  itemsize=itemsize,
                  cname=cname,
                  clevel=clevel,
                  use_dict=use_dict,
                  cnthreads=cnthreads,
                  dnthreads=dnthreads,
                  blocksize=blocksize,
                  filters=filters)

    if pshape is not None:
        assert a.pshape == pshape
    assert a.shape == shape
    assert a.itemsize == itemsize
    assert a.cname == cname
    assert a.clevel == clevel
    assert a.filters[-len(filters):] == filters

@pytest.mark.parametrize("shape, pshape, dtype, cname, clevel, use_dict, cnthreads, dnthreads, blocksize, filters",
                         [
                             ((100, 1230), (20, 10), np.float32, "lz4", 4, 0, 1, 1, 0, [1]),
                             ((23, 34), None, np.int64, "lz4hc", 8, 0, 2, 1, 256 * 1024, [2, 1]),
                             ((400, 399, 401), (20, 10, 130), np.int8, "blosclz", 5, 1, 2, 2, 128 * 1024, [4, 3, 0, 2, 1])
                         ])
def test_empty_numpy(shape, pshape, dtype, cname, clevel, use_dict, cnthreads, dnthreads, blocksize, filters):
    a = cat.empty(shape, pshape=pshape,
                  dtype=dtype,
                  cname=cname,
                  clevel=clevel,
                  use_dict=use_dict,
                  cnthreads=cnthreads,
                  dnthreads=dnthreads,
                  blocksize=blocksize,
                  filters=filters)

    if pshape is not None:
        assert a.pshape == pshape
    assert a.shape == shape
    assert a.dtype == dtype
    assert a.itemsize == np.dtype(dtype).itemsize
    assert a.cname == cname
    assert a.clevel == clevel
    assert a.filters[-len(filters):] == filters
