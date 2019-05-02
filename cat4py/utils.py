import cat4py as cat
import numpy as np


def fromnumpy(a, pshape=None, cparams=None, dparams=None, filename=None): #TODO: Add copy and mutable flags
    ndim = a.ndim
    if pshape is not None:
        assert (ndim == len(pshape))
    b = cat.Container(pshape=pshape, cparams=cparams, dparams=dparams, filename=filename)
    b.frombuffer(a.shape, bytes(a))
    return b


def tonumpy(a, dtype=np.float32):
    return np.frombuffer(a.tobuffer(), dtype=dtype).reshape(a.shape)
