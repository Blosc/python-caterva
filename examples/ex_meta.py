import cat4py as cat
import numpy as np
import os


shape = (128, 128)
chunks = (32, 32)
blocks = (8, 8)

urlpath = "ex_meta.cat"
if os.path.exists(urlpath):
    # Remove file on disk
    os.remove(urlpath)

dtype = np.dtype(np.complex128)
itemsize = dtype.itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

meta = {
    "m1": b"1111",
    "m2": b"2222",
}
# Create a caterva array from a numpy array (on disk)
a = cat.from_buffer(bytes(nparray), nparray.shape, chunks=chunks, blocks=blocks,
                    urlpath=urlpath, itemsize=itemsize, meta=meta)

# Read a caterva array from disk
b = cat.open(urlpath)

# Deal with meta
m1 = b.meta.get("m5", b"0000")
m2 = b.meta["m2"]

# Remove file on disk
os.remove(urlpath)
