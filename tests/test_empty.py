import cat4py as cat
import pytest


@pytest.mark.parametrize("shape, pshape, itemsize, compcode, clevel, use_dict, cnthreads, dnthreads, blocksize, filters",
                         [
                             ((100, 1230), (20, 10), 4, 1, 4, 0, 1, 1, 0, [1]),
                             ((23, 34), None, 8, 2, 8, 0, 2, 1, 256 * 1024, [2, 1]),
                             ((400, 399, 401), (20, 10, 130), 3, 4, 5, 1, 2, 2, 128 * 1024, [4, 3, 0, 2, 1])
                         ])
def test_empty(shape, pshape, itemsize, compcode, clevel, use_dict, cnthreads, dnthreads, blocksize, filters):
    a = cat.empty(shape, pshape=pshape,
                  itemsize=itemsize,
                  compcode=compcode,
                  clevel=clevel,
                  use_dict=use_dict,
                  cnthreads=cnthreads,
                  dnthreads=dnthreads,
                  blocksize=blocksize,
                  filters=filters)

    assert a.pshape == pshape
    assert a.shape == shape
    assert a.itemsize == itemsize
    assert a.compcode == compcode
    assert a.clevel == clevel
    assert a.filters[-len(filters):] == filters
