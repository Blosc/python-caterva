from . import container_ext as ext
import numpy as np
from .container import Container, process_key


class ReadIter(ext.ReadIter):
    def __init__(self, arr, itershape=None):
        super(ReadIter, self).__init__(arr, itershape)


class WriteIter(ext.WriteIter):
    def __init__(self, arr):
        super(WriteIter, self).__init__(arr)


class TLArray(Container):
    def __init__(self, **kwargs):
        """The basic, multidimensional and type-less data container.

        As this inherits from the :py:class:`Container` class, you can pass any
        keyword argument that is supported by the :py:meth:`Container.__init__`
        constructor.
        """
        self.pre_init(**kwargs)
        super(TLArray, self).__init__(**self.kwargs)

    def pre_init(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def cast(cls, cont):
        assert isinstance(cont, Container)
        cont.__class__ = cls
        assert isinstance(cont, TLArray)
        return cont

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
        buff = super(TLArray, self).__getitem__(key)
        return buff

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
        """Copy into a new container whose properties are specified in `kwargs`.

        Returns
        -------
        TLArray
            A new TLArray container that contains the copy.
        """
        arr = TLArray(**kwargs)
        return super(TLArray, self).copy(arr, **kwargs)

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
