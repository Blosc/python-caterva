from .container_ext import _Container, _getitem, _setitem, _copy, _from_file, _to_buffer, _from_buffer
import numpy as np


class Container(_Container):

    def __init__(self, pshape=None, filename=None, **kargs):
        self.kargs = kargs
        super(Container, self).__init__(pshape, filename, **kargs)

    def __getitem__(self, key):
        arr = Container(pshape=None, **self.kargs)
        _getitem(self, arr, key)
        return arr

    def __setitem__(self, key, item):
        _setitem(self, key, item)

    def copy(self, pshape=None, filename=None):
        arr = Container(pshape=pshape, filename=filename, **self.kargs)
        _copy(self, arr)
        return arr

    def to_buffer(self):
        return _to_buffer(self)

    def from_buffer(self, shape, buf):
        _from_buffer(self, shape, buf)

    def to_numpy(self, dtype):
        return np.frombuffer(self.to_buffer(), dtype=dtype).reshape(self.shape)

    def from_numpy(self, array):
        self.from_buffer(array.shape, bytes(array))


def from_file(filename):
    arr = Container()
    _from_file(arr, filename)
