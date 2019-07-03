from .container_ext import _Container, _getitem
import numpy as np


class Container(_Container):

    def __init__(self, pshape=None, filename=None, **kargs):
        self.kargs = kargs
        super(Container, self).__init__(pshape, filename, **kargs)

    def __getitem__(self, key):
        a = Container(pshape=None, **self.kargs)
        _getitem(self, a, key)
        return a


    def to_numpy(self, dtype):
        return np.frombuffer(self.to_buffer(), dtype=dtype).reshape(self.shape)

    def from_numpy(self, array):
        self.from_buffer(array.shape, bytes(array))
