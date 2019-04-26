import struct
import cat4py as cat
import numpy as np

cparams = cat.CParams(itemsize=4)  # we will be dealing with 4's
dparams = cat.DParams()
ctx = cat.Context(cparams, dparams)
print(ctx)

a = cat.Container(ctx, (200, 300))
print(a)

# Fill the array with 6 * 4 partitions with int32 4's
a.fill((4 * 200, 6 * 300), struct.pack("i", 4))

print("Compression ratio", a.cratio)
print(f"Shape: {a.shape}")

b = a.to_numpy(np.int32)

print(b)
