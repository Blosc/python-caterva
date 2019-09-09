from . import container_ext as ext
import numpy as np
import msgpack


class _ReadIter(ext.ReadIter):
    def __init__(self, arr, blockshape):
        super(_ReadIter, self).__init__(arr, blockshape)


class _WriteIter(ext.WriteIter):
    def __init__(self, arr):
        super(_WriteIter, self).__init__(arr)


def process_key(key, ndim):
    if not isinstance(key, (tuple, list)):
        key = (key,)
    key = tuple(k if isinstance(k, slice) else slice(k, k + 1) for k in key)
    if len(key) < ndim:
        key += tuple(slice(None) for _ in range(ndim - len(key)))
    return key


class Container(ext.Container):

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
            The index for the slices to be updated.
            Note that `step` parameter is not honored yet in slices.

        Returns
        -------
        bytes
            The a buffer with the requested data.
        """
        key = process_key(key, self.ndim)
        buff = ext.getitem(self, key)
        return buff

    def __setitem__(self, key, item):
        """Set a (multidimensional) slice as specified in `key`.

        Currently, this only works on containers backed by a plain buffer
        (i.e. pshape == None).

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated.
            Note that `step` parameter is not honored yet in slices.
        item: bytes
            The buffer with the values to be used for the update.
        """
        key = process_key(key, self.ndim)
        ext.setitem(self, key, item)

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
        return _ReadIter(self, blockshape)

    def iter_write(self):
        """Iterate over non initialized data blocks.

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
        return _WriteIter(self)

    def copy(self, **kwargs):
        """Copy to a new container whose properties are specified in `kwargs`.

        Returns
        -------
        Container
            A new container that contains the copy.
        """
        arr = Container(**kwargs)
        ext.copy(self, arr)
        return arr

    def to_buffer(self):
        """Return a buffer with the data contents.

        Returns
        -------
        bytes
            The buffer containing the data of the whole Container.
        """
        return ext.to_buffer(self)

    def to_numpy(self, dtype):
        """Return a NumPy array with the data contents and `dtype`.

        Parameters
        ----------
        dtype: a dtype instance or string
            The dtype for the returned NumPy array.

        Returns
        -------
        ndarray
            The NumPy array object containing the data of the whole Container.
        """
        return np.frombuffer(self.to_buffer(), dtype=dtype).reshape(self.shape)

    def has_metalayer(self, name):
        """Whether `name` is an existing metalayer or not.

        Parameters
        ----------
        name: str
            The name of the metalayer to check.

        Returns
        -------
        bool
            True if metalayer exists in `self`; else False.
        """
        return ext.has_metalayer(self, name)

    def get_metalayer(self, name):
        """Return the `name` metalayer.

        Parameters
        ----------
        name: str
            The name of the metalayer to return.

        Returns
        -------
        bytes
            The buffer containing the metalayer info (typically in msgpack
            format).
        """
        if self.has_metalayer(name) is False:
            return None
        content = ext.get_metalayer(self, name)
        return msgpack.unpackb(content)

    def update_metalayer(self, name, content):
        """Update the `name` metalayer with `content`.

        Parameters
        ----------
        name: str
            The name of the metalayer to update.
        content: bytes
            The buffer containing the new content for the metalayer.
            Note that the *length* of the metalayer cannot not change,
            else an exception will be raised.

        """
        content = msgpack.packb(content)
        return ext.update_metalayer(self, name, content)

    def get_usermeta(self):
        """Return the `usermeta` section.

        Returns
        -------
        bytes
            The buffer for the usermeta section (typically in msgpack format,
            but not necessarily).
        """
        content = ext.get_usermeta(self)
        return msgpack.unpackb(content)

    def update_usermeta(self, content):
        """Update the `usermeta` section.

        Parameters
        ----------
        content: bytes
            The buffer containing the new `usermeta` data that replaces the
            previous one.  Note that the length of the new content can be
            different from the existing one.

        """
        content = msgpack.packb(content)
        return ext.update_usermeta(self, content)


def empty(shape, **kwargs):
    """Create an empty container.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final container.

    In addition, you can pass any keyword argument that is supported by the
    `Container` class.

    Returns
    -------
    Container
        The new Container object.
    """
    arr = Container(**kwargs)
    arr.updateshape(shape)
    return arr


def from_buffer(buffer, shape, **kwargs):
    """Create a container out of a buffer.

    Parameters
    ----------
    buffer: bytes
        The buffer of the data to populate the container.
    shape: tuple or list
        The shape for the final container.

    In addition, you can pass any keyword argument that is supported by the
    `Container` class.

    Returns
    -------
    Container
        The new Container object.
    """
    arr = Container(**kwargs)
    ext.from_buffer(arr, shape, buffer)
    return arr


def from_numpy(nparray, **kwargs):
    """Create a container out of a NumPy array.

    Parameters
    ----------
    nparray: NumPy array
        The NumPy array to populate the container with.

    In addition, you can pass any keyword argument that is supported by the
    `Container` class.

    Returns
    -------
    Container
        The new Container object.
    """
    arr = from_buffer(bytes(nparray), nparray.shape,
                      itemsize=nparray.itemsize, **kwargs)
    return arr


def from_file(filename):
    """Open a new container from `filename`.

    Parameters
    ----------
    filename: str
        The filename where the data is.  The file should have a Blosc2 frame
        with a Caterva metalayer on it.

    In addition, you can pass any keyword argument that is supported by the
    `Container` class.

    Returns
    -------
    Container
        The new Container object.
    """
    arr = Container()
    ext.from_file(arr, filename)
    # if arr.has_metalayer("numpy"):
    #     arr.__class__ = Array
    return arr
