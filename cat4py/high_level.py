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

    def __init__(self, pshape=None, filename=None, **kwargs):
        """Multidimensional and type-less data container.

        Parameters
        ----------
        pshape: iterable object or None
            The partition shape.  If None, the store is a plain buffer (non-compressed).
        filename: str or None
            The name of the file to store data.  If `None`, data store is in-memory.
        kwargs: dict
            Optional parameters for compression and decompression.  Also:
            metalayers: dict or None
                A dictionary with different metalayers.  One entry per metalayer:
                    key: bytes or str
                        The name of the metalayer.
                    value: object
                        The metalayer object that will be (de-)serialized using msgpack.
        """
        super(Container, self).__init__(pshape, filename, **kwargs)

    def __getitem__(self, key):
        if not isinstance(key, (tuple, list)):
             key = (key,)
        key = tuple(k if isinstance(k, slice) else slice(k, k + 1) for k in key)
        if len(key) < self.ndim:
            key += tuple(slice(None) for i in range(self.ndim - len(key)))
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

    def update_metalayer(self, name, content):
        content = msgpack.packb(content)
        return ext._update_metalayer(self, name, content)

    def get_usermeta(self):
        content = ext._get_usermeta(self)
        return msgpack.unpackb(content)

    def update_usermeta(self, content):
        content = msgpack.packb(content)
        return ext._update_usermeta(self, content)


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
