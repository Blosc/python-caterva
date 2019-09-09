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

    def __init__(self, **kwargs):
        """The basic and multidimensional and type-less data container.

        Parameters
        ----------
        kwargs: dict
            Optional parameters for compression and decompression.  Also:
            pshape: iterable object or None
                The partition shape.  If None, the store is a plain buffer (non-compressed).
            filename: str or None
                The name of the file to store data.  If `None`, data is stores in-memory.
            metalayers: dict or None
                A dictionary with different metalayers.  One entry per metalayer:
                    key: bytes or str
                        The name of the metalayer.
                    value: object
                        The metalayer object that will be (de-)serialized using msgpack.
        """
        super(Container, self).__init__(**kwargs)

    def __getitem__(self, key):
        """Return a (multidimensional) slice as specified in `key`.

        Parameters
        ----------
        key: int, slice or sequence of slices
            Note that `step` parameter is not honored yet in slices.

        Returns
        -------
        bytes
            The a buffer with the requested data.
        """
        if not isinstance(key, (tuple, list)):
             key = (key,)
        key = tuple(k if isinstance(k, slice) else slice(k, k + 1) for k in key)
        if len(key) < self.ndim:
            key += tuple(slice(None) for i in range(self.ndim - len(key)))
        buff = ext._getitem(self, key)
        return buff

    def __setitem__(self, key, item):
        """Set a (multidimensional) slice as specified in `key`.

        Currently, this only works on containers backed by a plain buffer
        (i.e. pshape == None).

        Parameters
        ----------
        key: int, slice or sequence of slices
            Note that `step` parameter is not honored yet in slices.
        """
        ext._setitem(self, key, item)

    def iter_read(self, blockshape):
        """Iterate over data blocks whose dims are specified in `blockshape`.

        Parameters
        ----------
        blockshape: tuple, list
            The shape in which the data block will be returned.

        Yields
        ------
        tuple of (block, info)
            block: bytes
                The buffer with the data block.
            info: namedtuple
                Info about the returned data block.  Its structure is:
                namedtuple("IterInfo", "slice, shape, size")
                IterInfo:
                    slice: tuple
                        The coordinates where the data block starts.
                    shape: tuple
                        The shape of the actual data block (it can be
                        smaller than `blockshape` at the edges of the array).
                    size: int
                        The size, in elements, of the block.
        """
        return ReadIter(self, blockshape)

    def iter_write(self):
        """Iterate over non initialized data blocks.

        Parameters
        ----------
        blockshape: tuple, list
            The shape in which the data block should be delivered for filling.

        Yields
        ------
        tuple of (block, info)
            block: bytes
                The buffer with the data block to be filled.
            info: namedtuple
                Info about the data block to be filled.  Its structure is:
                namedtuple("IterInfo", "slice, shape, size")
                IterInfo:
                    slice: tuple
                        The coordinates where the data block starts.
                    shape: tuple
                        The shape of the actual data block (it can be
                        smaller than `blockshape` at the edges of the array).
                    size: int
                        The size, in elements, of the block.
        """
        return WriteIter(self)

    def copy(self, **kwargs):
        """Copy to a new container whose properties are specified in `kwargs`.

        Returns
        -------
        Container
            The new container that contains the copy.
        """
        arr = Container(**kwargs)
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


def empty(shape, **kwargs):
    arr = Container(**kwargs)
    arr.updateshape(shape)
    return arr


def from_buffer(buffer, shape, **kwargs):
    arr = Container(**kwargs)
    ext._from_buffer(arr, shape, buffer)
    return arr


def from_numpy(nparray, **kwargs):
    arr = from_buffer(bytes(nparray), nparray.shape, **kwargs)
    return arr


def from_file(filename):
    arr = Container()
    ext._from_file(arr, filename)
    # if arr.has_metalayer("numpy"):
    #     arr.__class__ = Array
    return arr
