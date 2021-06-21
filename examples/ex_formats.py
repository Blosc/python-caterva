import numpy as np
import cat4py as cat
from time import time
import os

urlpath_sparse = "ex_formats_sparse.caterva"
# urlpath_sparse = None
urlpath_sequential = "ex_formats_sequential.caterva"
# urlpath_sequential = None

if urlpath_sparse and os.path.exists(urlpath_sparse):
    cat.remove(urlpath_sparse)

if urlpath_sequential and os.path.exists(urlpath_sequential):
    cat.remove(urlpath_sequential)

shape = (1000 * 1000,)
chunks = (100,)
blocks = (100,)
dtype = np.dtype(np.float64)
itemsize = dtype.itemsize

t0 = time()
a = cat.empty(shape, 8, chunks=chunks, blocks=blocks, urlpath=urlpath_sparse,
             sequential=False)
for nchunk in range(a.nchunks):
    a[nchunk * chunks[0]: (nchunk + 1) * chunks[0]] = np.arange(chunks[0], dtype=dtype)
t1 = time()

print(f"Time: {(t1 - t0):.4f} s")
print(a.nchunks)
an = np.array(a[:]).view(dtype)


t0 = time()
b = cat.empty(shape, itemsize=itemsize, chunks=chunks, blocks=blocks, urlpath=urlpath_sequential, sequential=True)

print(b.nchunks)
for nchunk in range(shape[0] // chunks[0]):
    b[nchunk * chunks[0]: (nchunk + 1) * chunks[0]] = np.arange(chunks[0], dtype=dtype)
t1 = time()

print(f"Time: {(t1 - t0):.4f} s")
print(b.nchunks)
bn = np.array(b[:]).view(dtype)

np.testing.assert_allclose(an, bn)
