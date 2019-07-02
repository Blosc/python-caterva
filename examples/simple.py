import struct
import cat4py as cat
import numpy as np

cparams = cat.CParams(itemsize=4, compcode=0)  # we will be dealing with itemsizes of 4's
dparams = cat.DParams()

# Create an array with partitions (backed with a C-Blosc2 super-chunk)
a = cat._Container(pshape=(200, 300), cparams=cparams, dparams=dparams)
# Create a plain-buffer array
b = cat._Container(pshape=None, cparams=cparams, dparams=dparams)

# Fill the arrays with 6 * 4 partitions with int32 4's
a.fill((4 * 200, 6 * 300), struct.pack("f", 4.45))
b.fill((4 * 200, 6 * 300), struct.pack("f", 4.45))

print(f"Shape for the created arrays: {a.shape}")
print("Block shape for the created arrays:")
print(f"- Container a: {a.pshape}")
print(f"- Container b: {b.pshape}")

assert(a.shape == b.shape)
print("Compression ratio for chunked array:", a.cratio)
print("Compression ratio for plain buffer array:", b.cratio)

# Convert back to a numpy array and compare results
c = np.frombuffer(a.to_buffer(), np.float32)
print(c)
d = np.frombuffer(b.to_buffer(), np.float32)

np.testing.assert_almost_equal(c, d)
