---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tutorial
Caterva functions let users to perform different operations with Caterva arrays like setting, copying or slicing them.
In this section, we are going to see how to create and manipulate a Caterva array in a simple way.

```{code-cell} ipython3
import caterva as cat

cat.__version__
```

## Creating an array
First, we create an array, with zero being used as the default value for uninitialized portions of the array.

```{code-cell} ipython3
c = cat.zeros((10000, 10000), itemsize=4, chunks=(1000, 1000), blocks=(100, 100))

c
```

## Reading and writing data
We can access and edit Caterva arrays using NumPy.

```{code-cell} ipython3
import struct
import numpy as np

dtype = np.int32

c[0, :] = np.arange(10000, dtype=dtype)
c[:, 0] = np.arange(10000, dtype=dtype)
```

```{code-cell} ipython3
c[0, 0]
```

```{code-cell} ipython3
np.array(c[0, 0]).view(dtype)
```

```{code-cell} ipython3
np.array(c[0, -1]).view(dtype)
```

```{code-cell} ipython3
np.array(c[0, :]).view(dtype)
```

```{code-cell} ipython3
np.array(c[:, 0]).view(dtype)
```

```{code-cell} ipython3
np.array(c[:]).view(dtype)
```

## Persistent data
When we create a Caterva array, we can we can specify where it will be stored.
Then, we can access to this array whenever we want and it will still contain all the data as it is stored persistently.

```{code-cell} ipython3
c1 = cat.full((1000, 1000), fill_value=b"pepe", chunks=(100, 100), blocks=(50, 50),
             urlpath="cat_tutorial.caterva")
```

```{code-cell} ipython3
c2 = cat.open("cat_tutorial.caterva")

c2.info
```

```{code-cell} ipython3
np.array(c2[0, 20:30]).view("S4")
```

```{code-cell} ipython3
import os
if os.path.exists("cat_tutorial.caterva"):
  cat.remove("cat_tutorial.caterva")
```

## Compression params
Here we can see how when we make a copy of a Caterva array we can change its compression parameters in an easy way. 

```{code-cell} ipython3
b = np.arange(1000000).tobytes()

c1 = cat.from_buffer(b, shape=(1000, 1000), itemsize=8, chunks=(500, 10), blocks=(50, 10))

c1.info
```

```{code-cell} ipython3
c2 = c1.copy(chunks=(500, 10), blocks=(50, 10),
             codec=cat.Codec.ZSTD, clevel=9, filters=[cat.Filter.BITSHUFFLE])

c2.info
```

## Metalayers
Metalayers are small metadata for informing about the properties of data that is stored on a container. 
The metalayers of a Caterva array are also easy to access and edit by users.

```{code-cell} ipython3
from msgpack import packb, unpackb
```

```{code-cell} ipython3
meta = {
    "dtype": packb("i8"),
    "coords": packb([5.14, 23.])
}
```

```{code-cell} ipython3
c = cat.zeros((1000, 1000), 5, chunks=(100, 100), blocks=(50, 50), meta=meta)
```

```{code-cell} ipython3
len(c.meta)
```

```{code-cell} ipython3
c.meta.keys()
```

```{code-cell} ipython3
for key in c.meta:
    print(f"{key} -> {unpackb(c.meta[key])}")
```

```{code-cell} ipython3
c.meta["coords"] = packb([0., 23.])
```

```{code-cell} ipython3
for key in c.meta:
    print(f"{key} -> {unpackb(c.meta[key])}")
```

## Small tutorial
In this example it is shown how easy is to create a Caterva array from an image and how users can manipulate it using Caterva and Image functions.  

```{code-cell} ipython3
from PIL import Image
```

```{code-cell} ipython3
im = Image.open("../_static/blosc-logo_128.png")

im
```

```{code-cell} ipython3
meta = {"dtype": b"|u1"}

c = cat.asarray(np.array(im), chunks=(50, 50, 4), blocks=(10, 10, 4), meta=meta)

c.info
```

```{code-cell} ipython3
im2 = c[15:55, 10:35]  # Letter B

Image.fromarray(np.array(im2).view(c.meta["dtype"]))
```

```{code-cell} ipython3

```
