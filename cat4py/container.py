from . import container_ext as ext
import numpy as np
import msgpack


class ReadIter(ext.ReadIter):
    def __init__(self, arr, blockshape=None):
        super(ReadIter, self).__init__(arr, blockshape)


class WriteIter(ext.WriteIter):
    def __init__(self, arr):
        super(WriteIter, self).__init__(arr)


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
        pshape: iterable object or None
            The partition shape.  If None, the store is a plain buffer (non-compressed).
        filename: str or None
            The name of the file to store data.  If `None`, data is stores in-memory.
        memframe: bool
            If True, the Container is backed by a frame in-memory.  Else, by a
            super-chunk.  Default: False.
        metalayers: dict or None
            A dictionary with different metalayers.  One entry per metalayer:
                key: bytes or str
                    The name of the metalayer.
                value: object
                    The metalayer object that will be (de-)serialized using msgpack.
        itemsize: int
            The number of bytes for the itemsize in container.  Default: 4.
        cname: string
            The name for the compressor codec.  Default: "lz4".
        clevel: int (0 to 9)
            The compression level.  0 means no compression, and 9 maximum compression.
            Default: 5.
        filters: list
            The filter pipeline.  Default: [cat4py.SHUFFLE]
        filters_meta: list
            The meta info for each filter in pipeline.  An uint8 per slot. Default: [0]
        cnthreads: int
            The number of threads for compression.  Default: 1.
        dnthreads: int
            The number of threads for decompression.  Default: 1.
        blocksize: int
            The blocksize for every chunk in container.  The default is 0 (automatic).
        use_dict: bool
            If a dictionary should be used during compression.  Default: False.

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
        buff = super(Container, self).__getitem__(key)
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
        super(Container, self).__setitem__(key, item)

    def iter_read(self, blockshape=None):
        """Iterate over data blocks whose dims are specified in `blockshape`.

        Parameters
        ----------
        blockshape: tuple, list
            The shape in which the data block will be returned.  If `None`,
            the `Container.pshape` will be used as `blockshape`.

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
            A new container that contains the copy.
        """
        arr = Container(**kwargs)
        super(Container, self).copy(arr)
        return arr

    def to_buffer(self):
        """Return a buffer with the data contents.

        Returns
        -------
        bytes
            The buffer containing the data of the whole Container.
        """
        return super(Container, self).to_buffer()

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
        # Alternate way to build a numpy array, but a bit slower
        # arr = np.empty(self.shape, dtype)
        # for block, info in self.iter_read(self.pshape):
        #     arr[info.slice] = np.frombuffer(block, dtype=dtype).reshape(info.shape)
        # return arr
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

    In addition to regular arguments, you can pass any keyword argument that
    is supported by the :py:meth:`Container.__init__` constructor.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final container.

    Returns
    -------
    Container
        The new :py:class:`Container` object.
    """
    arr = Container(**kwargs)
    arr.updateshape(shape)
    return arr


def from_buffer(buffer, shape, **kwargs):
    """Create a container out of a buffer.

    In addition to regular arguments, you can pass any keyword argument that
    is supported by the :py:meth:`Container.__init__` constructor.

    Parameters
    ----------
    buffer: bytes
        The buffer of the data to populate the container.
    shape: tuple or list
        The shape for the final container.

    Returns
    -------
    Container
        The new :py:class:`Container` object.
    """
    arr = Container(**kwargs)
    ext.from_buffer(arr, shape, buffer)
    return arr


def from_numpy(nparray, **kwargs):
    """Create a container out of a NumPy array.

    In addition to regular arguments, you can pass any keyword argument that
    is supported by the :py:meth:`Container.__init__` constructor.

    Parameters
    ----------
    nparray: NumPy array
        The NumPy array to populate the container with.

    Returns
    -------
    Container
        The new :py:class:`Container` object.
    """
    arr = from_buffer(bytes(nparray), nparray.shape,
                      itemsize=nparray.itemsize, **kwargs)
    return arr


def from_file(filename, copy=False):
    """Open a new container from `filename`.

    Parameters
    ----------
    filename: str
        The file having a Blosc2 frame format with a Caterva metalayer on it.
    copy: bool
        If true, the container is backed by a new, sparse in-memory super-chunk.
        Else, an on-disk, frame-backed one is created (i.e. no copies are made).

    Returns
    -------
    Container
        The new :py:class:`Container` object.
    """
    arr = Container()
    ext.from_file(arr, filename, copy)
    # if arr.has_metalayer("numpy"):
    #     arr.__class__ = Array
    return arr


def from_sframe(sframe, copy=False):
    """Open a new container from `sframe`.

    Parameters
    ----------
    sframe: bytes
        The Blosc2 serialized frame with a Caterva metalayer on it.
    copy: bool
        If true, the container is backed by a new, sparse in-memory super-chunk.
        Else, an in-memory, frame-backed one is created (i.e. no copies are made).

    Returns
    -------
    Container
        The new :py:class:`Container` object.
    """
    arr = Container()
    ext.from_sframe(arr, sframe, copy)
    # if arr.has_metalayer("numpy"):
    #     arr.__class__ = Array
    return arr
