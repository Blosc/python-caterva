# This module is only useful to been able to change the Container class to different subclasses.
# This can't be done on a Cython extension (ext.Container).


import msgpack
from . import container_ext as ext
import numpy as np


def process_key(key, ndim):
    if not isinstance(key, (tuple, list)):
        key = (key,)
    key = tuple(k if isinstance(k, slice) else slice(k, k + 1) for k in key)
    if len(key) < ndim:
        key += tuple(slice(None) for _ in range(ndim - len(key)))
    return key


class ReadIter(ext.ReadIter):
    def __init__(self, arr, itershape=None):
        super(ReadIter, self).__init__(arr, itershape)


class WriteIter(ext.WriteIter):
    def __init__(self, arr):
        super(WriteIter, self).__init__(arr)


class NDArray(ext.Container):
    def __init__(self, **kwargs):
        """The low-level, multidimensional and type-less data container.

        Parameters
        ----------
        chunkshape: iterable object or None
            The chunk shape.  If None, the store is a plain buffer (non-compressed).
        blockshape: iterable object or None
            The block shape.  If None, the store is a plain buffer (non-compressed).
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
        filtersmeta: list
            The meta info for each filter in pipeline.  An uint8 per slot. Default: [0]
        nthreads: int
            The number of threads.  Default: 1.
        usedict: bool
            If a dictionary should be used during compression.  Default: False.
        """
        if type(self) == NDArray:
            self.pre_init(**kwargs)
        super(NDArray, self).__init__(**self.kwargs)

    def pre_init(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def cast(cls, cont):
        cont.__class__ = cls
        assert isinstance(cont, NDArray)
        return cont

    @property
    def __array_interface__(self):
        print("Array interface")
        interface = {
            "data": self,
            "shape": self.shape,
            "typestr": f'S{self.itemsize}',
            "version": 3
        }
        return interface

    def __getitem__(self, item):
        return self.slice(item)

    def slice(self, item, **kwargs):
        arr = NDArray(**kwargs)
        return ext.get_slice(arr, self, process_key(item, self.ndim), **kwargs)


    def iter_read(self, itershape=None):
        """Iterate over data blocks whose dims are specified in `blockshape`.

        Parameters
        ----------
        itershape: tuple, list
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
        return ReadIter(self, itershape)

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

    def squeeze(self, **kwargs):
        """Remove the 1's in Container's shape."""
        super(NDArray, self).squeeze(**kwargs)

    def to_buffer(self, **kwargs):
        """Returns a buffer with the data contents.

        Returns
        -------
        bytes
            The buffer containing the data of the whole Container.
        """
        return super(NDArray, self).to_buffer(**kwargs)

    def to_sframe(self, **kwargs):
        """Return a serialized frame with data and metadata contents.

        Returns
        -------
        bytes or MemoryView
            A buffer containing a serial version of the whole Container.
            When the Container is backed by an in-memory frame, a MemoryView
            of it is returned.  If not, a bytes object with the frame is
            returned.
        """
        return super(NDArray, self).to_sframe(**kwargs)

    def copy(self, **kwargs):
        """Copy into a new container.

        Returns
        -------
        NDArray
            The `arr` object containing the copy.
        """
        arr = NDArray(**kwargs)
        return ext.copy(arr, self, **kwargs)

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
        return super(NDArray, self).has_metalayer(name)

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
        content = super(NDArray, self).get_metalayer(name)

        return msgpack.unpackb(content, raw=True)

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
        return super(NDArray, self).update_metalayer(name, content)

    def get_usermeta(self):
        """Return the `usermeta` section.

        Returns
        -------
        bytes
            The buffer for the usermeta section (typically in msgpack format,
            but not necessarily).
        """
        content = super(NDArray, self).get_usermeta()
        return msgpack.unpackb(content, raw=True)

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
        return super(NDArray, self).update_usermeta(content)

    def to_numpy(self, dtype):
        """Returns a NumPy array with the data contents and `dtype`.

        Returns
        -------
        numpy.array
            The NumPy array object containing the data of the whole Container.
        """
        return np.fromstring(self.to_buffer(), dtype=dtype).reshape(self.shape)