import numpy as np
import cat4py as cat
import pytest


@pytest.mark.parametrize("shape, chunkshape, itemsize, compname, complevel, use_dict, nthreads, blocksize, filters",
                         [
                             ((100, 1230), (20, 10), 4, "lz4", 4, 0, 1, 0, [1]),
                             ((23, 34), None, 8, "lz4hc", 8, 0, 2, 256 * 1024, [2, 1]),
                             ((400, 399, 401), (20, 10, 130), 3, "blosclz", 5, 1, 2, 128 * 1024, [4, 3, 0, 2, 1])
                         ])
def test_empty(shape, chunkshape, itemsize, compname, complevel, use_dict, nthreads, blocksize, filters):
    a = cat.empty(shape, chunkshape=chunkshape,
                  itemsize=itemsize,
                  compname=compname,
                  complevel=complevel,
                  use_dict=use_dict,
                  nthreads=nthreads,
                  blocksize=blocksize,
                  filters=filters)

    if chunkshape is not None:
        assert a.chunkshape == chunkshape
    assert a.shape == shape
    assert a.itemsize == itemsize
    assert a.compname == (compname if chunkshape is not None else None)
    assert a.complevel == (complevel if chunkshape is not None else 1)
    if chunkshape is not None:
        assert a.filters[-len(filters):] == filters
    else:
        assert a.filters is None


@pytest.mark.parametrize("shape, chunkshape, dtype, compname, complevel, use_dict, nthreads, blocksize, filters",
                         [
                             ((100, 1230), (20, 10), np.float32, "lz4", 4, 0, 1, 0, [1]),
                             ((23, 34), None, np.int64, "lz4hc", 8, 0, 2, 256 * 1024, [2, 1]),
                             ((400, 399, 401), (20, 10, 130), np.int8, "blosclz", 5, 1, 2, 128 * 1024, [4, 3, 0, 2, 1])
                         ])
def test_empty_numpy(shape, chunkshape, dtype, compname, complevel, use_dict, nthreads, blocksize, filters):
    a = cat.empty(shape, chunkshape=chunkshape,
                  dtype=dtype,
                  compname=compname,
                  complevel=complevel,
                  use_dict=use_dict,
                  nthreads=nthreads,
                  blocksize=blocksize,
                  filters=filters)

    if chunkshape is not None:
        assert a.chunkshape == chunkshape
    assert a.shape == shape
    assert a.dtype == dtype
    assert a.itemsize == np.dtype(dtype).itemsize
    assert a.compname == (compname if chunkshape is not None else None)
    assert a.complevel == (complevel if chunkshape is not None else 1)
    if chunkshape is not None:
        assert a.filters[-len(filters):] == filters
    else:
        assert a.filters is None
