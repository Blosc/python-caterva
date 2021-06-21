import numpy as np
import caterva as cat
import pytest


@pytest.mark.parametrize("shape, chunks, blocks, itemsize, cname, clevel, use_dict, nthreads",
                         [
                             ((100, 1230), (200, 100), (55, 3), 4, cat.Codec.ZSTD, 4, 0, 1),
                             ((23, 34), None, None, 8, cat.Codec.BLOSCLZ, 8, 0, 2),
                             ((80, 51, 60), (20, 10, 33), (6, 6, 26), 3, cat.Codec.LZ4, 5, 1, 2)
                         ])
def test_zeros(shape, chunks, blocks, itemsize, cname, clevel, use_dict, nthreads):
    a = cat.zeros(shape, chunks=chunks,
                  blocks=blocks,
                  itemsize=itemsize,
                  cname=cname,
                  clevel=clevel,
                  use_dict=use_dict,
                  nthreads=nthreads)

    for i in np.nditer(np.array(a[:])):
        assert i == bytes(itemsize)
