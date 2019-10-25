from cat4py import container_ext as ext
import numpy as np

from .container import Container, process_key


class ReadIter(ext.ReadIter):
    def __init__(self, arr, blockshape):
        self.arr = arr
        super(ReadIter, self).__init__(arr, blockshape)

    def __next__(self):
        buff, info = ext.ReadIter.__next__(self)
        arr = np.frombuffer(buff, dtype=self.arr.dtype).reshape(info.shape)
        return arr, info


class WriteIter(ext.WriteIter):
    def __init__(self, arr):
        self.arr = arr
        super(WriteIter, self).__init__(arr)

    def __next__(self):
        buff, info = ext.WriteIter.__next__(self)
        arr = np.frombuffer(buff, dtype=self.arr.dtype).reshape(info.shape)
        return arr, info


class NPArray(Container):

    def __init__(self, dtype, **kwargs):
        """The multidimensional data container that plays well with NumPy.

        As this inherits from the :py:class:`Container` class, you can pass any
        keyword argument that is supported by the :py:meth:`Container.__init__`
        constructor, plus the `dtype`.

        Parameters
        ----------
        dtype: numpy.dtype
            The data type for the container.
        """
        self.dtype = np.dtype(dtype)
        self.kwargs = kwargs
        self.pre_init(self.dtype, **kwargs)
        super(NPArray, self).__init__(**self.kwargs)

    def pre_init(self, dtype, **kwargs):
        self.dtype = np.dtype(dtype)
        kwargs["itemsize"] = self.dtype.itemsize
        kwargs["metalayers"] = {"numpy": {
            # TODO: adding "version" does not deserialize well
            # "version": 0,    # can be any number up to 127
            "dtype": str(self.dtype),
        }}
        self.kwargs = kwargs

    @classmethod
    def cast(cls, cont):
        assert isinstance(cont, Container)
        cont.__class__ = cls
        assert isinstance(cont, NPArray)
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
        numpy.array
            The NumPy array with the requested data.
        """
        key = process_key(key, self.ndim)
        buff = super(NPArray, self).__getitem__(key)

        # shape = [k.stop - k.start for k in key]   # not quite correct
        # Trick to get the slice easily and without a lot of memory consumption
        # Maybe there are more elegant ways for this, but meanwhile ...
        a = np.lib.stride_tricks.as_strided(np.empty(0), self.shape, (0,) * len(self.shape))
        shape = a[key].shape
        return np.frombuffer(buff, dtype=self.dtype).reshape(shape)

    def __array__(self):
        """Convert into a NumPy object via the array protocol."""
        return self.to_numpy()

    def iter_read(self, blockshape=None):
        """Iterate over data blocks whose dims are specified in `blockshape`.

        Parameters
        ----------
        blockshape: tuple, list
            The shape in which the data block will be returned.  If `None`,
            the `NPContainer.pshape` will be used as `blockshape`.

        Yields
        ------
        tuple of (block, info)
            block: NumPy.array
                The numpy array with the data block.
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
            block: numpy.array
                The NumPy array with the data block to be filled.
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
        NPArray
            A new NPArray container that contains the copy.
        """
        arr = NPArray(self.dtype, **kwargs)
        return super(NPArray, self).copy(arr)

    def to_numpy(self):
        """Returns a NumPy array with the data contents and `dtype`.

        Returns
        -------
        numpy.array
            The NumPy array object containing the data of the whole Container.
        """
        return np.frombuffer(self.to_buffer(), dtype=self.dtype).reshape(self.shape)
