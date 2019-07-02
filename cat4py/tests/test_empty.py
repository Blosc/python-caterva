import cat4py as cat
import pytest


kargs = ['itemsize', 'compcode', 'clevel', 'use_dict', 'cnthreads', 'dnthreads', 'blocksize', 'filters']

@pytest.mark.parametrize("pshape, itemsize, compcode, clevel, use_dict, cnthreads, dnthreads, blocksize, filters",
                         [
                             ([20, 10], 4, 1, 4, 0, 1, 1, 0, [1]),
                             (None, 8, 2, 8, 0, 2, 1, 256 * 1024, [2, 1]),
                             ([20, 10, 130], 3, 4, 5, 1, 2, 2, 128 * 1024, [4, 3, 0, 2, 1])
                         ])
def test_empty(pshape, itemsize, compcode, clevel, use_dict, cnthreads, dnthreads, blocksize, filters):
    a = cat.Container(pshape=pshape,
                      itemsize=itemsize,
                      compcode=compcode,
                      clevel=clevel,
                      use_dict=use_dict,
                      cnthreads=cnthreads,
                      dnthreads=dnthreads,
                      blocksize=blocksize,
                      filters=filters)
    assert a.itemsize == itemsize
    assert a.compcode == compcode
    assert a.clevel == clevel
    assert a.filters[-len(filters):] == filters
