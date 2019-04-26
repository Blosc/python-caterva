import struct
import cat4py as cat
import numpy as np

cparams = cat.CParams(itemsize=4)  # we will be dealing with 4's
dparams = cat.DParams()
ctx = cat.Context(cparams, dparams)

a = cat.Container(ctx)

# Fill the array with 6 * 4 partitions with int32 4's
a.fill((4 * 200, 6 * 300), struct.pack("f", 4.45))

print("Compression ratio", a.cratio)
print(f"Shape: {a.shape}")

b = a.to_numpy(np.float32)

print(b)
