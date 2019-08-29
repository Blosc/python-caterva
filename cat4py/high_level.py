from . import container_ext as ext
import numpy as np
import msgpack


class ReadIter(ext._ReadIter):
    def __init__(self, arr, blockshape):
        super(ReadIter, self).__init__(arr, blockshape)


class WriteIter(ext._WriteIter):
    def __init__(self, arr):
        super(WriteIter, self).__init__(arr)


class Container(ext._Container):

    def __init__(self, pshape=None, filename=None, **kargs):
        self.kargs = kargs
        super(Container, self).__init__(pshape, filename, **kargs)

    def __getitem__(self, key):
        buff = ext._getitem(self, key)
        return buff

    def __setitem__(self, key, item):
        ext._setitem(self, key, item)

    def iter_read(self, blockshape):
        return ReadIter(self, blockshape)

    def iter_write(self):
        return WriteIter(self)

    def copy(self, pshape=None, filename=None, **kargs):
        arr = Container(pshape, filename, **kargs)
        ext._copy(self, arr)
        return arr

    def to_buffer(self):
        return ext._to_buffer(self)

    def to_numpy(self, dtype):
        return np.frombuffer(self.to_buffer(), dtype=dtype).reshape(self.shape)

    def has_metalayer(self, name):
        return ext._has_metalayer(self, name)

    def get_metalayer(self, name):
        if self.has_metalayer(name) is False:
            return None
        content = ext._get_metalayer(self, name)
        return msgpack.unpackb(content)

    def update_metalayer(self, name, dict):
        content = msgpack.packb(dict)
        return ext._update_metalayer(self, name, content)


def empty(shape, pshape=None, filename=None, **kargs):
    arr = Container(pshape, filename, **kargs)
    arr.updateshape(shape)
    return arr


def from_buffer(buffer, shape, pshape=None, filename=None, **kargs):
    arr = Container(pshape, filename, **kargs)
    ext._from_buffer(arr, shape, buffer)
    return arr


def from_numpy(nparray, pshape=None, filename=None, **kargs):
    arr = from_buffer(bytes(nparray), nparray.shape, pshape, filename, **kargs)
    return arr


def from_file(filename):
    arr = Container()
    ext._from_file(arr, filename)
    # if arr.has_metalayer("numpy"):
    #     arr.__class__ = Array
    return arr
