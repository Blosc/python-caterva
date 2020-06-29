from cat4py import container_ext as ext
import numpy as np

from .ndarray import NDArray, process_key


class ReadIter(ext.ReadIter):
    def __init__(self, arr, itershape):
        self.arr = arr
        super(ReadIter, self).__init__(arr, itershape)

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


class NDTArray(NDArray):

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
        if type(self) == NDTArray:
            self.pre_init(dtype, **kwargs)
        super(NDTArray, self).__init__(**self.kwargs)

    def pre_init(self, dtype, **kwargs):
        self.dtype = np.dtype(dtype)
        kwargs["itemsize"] = self.dtype.itemsize
        kwargs["metalayers"] = {"type": {
            # TODO: adding "version" does not deserialize well
            # "version": 0,    # can be any number up to 127
            "dtype": str(self.dtype),
            }
        }
        self.kwargs = kwargs

    @classmethod
    def cast(cls, cont):
        cont.__class__ = cls
        assert isinstance(cont, NDTArray)
        return cont

    def __getitem__(self, item):
        return self.slice(item)

    def slice(self, item, **kwargs):
        arr = NDTArray(self.dtype, **kwargs)
        return ext.get_slice(arr, self, process_key(item, self.ndim), **kwargs)

    @property
    def __array_interface__(self):
        print("Array interface")
        interface = {
            "data": self,
            "shape": self.shape,
            "typestr": str(self.dtype),
            "version": 3
        }
        return interface


    def copy(self, **kwargs):
        """Copy into a new container whose properties are specified in `kwargs`.

        Returns
        -------
        NDTArray
            A new NPArray container that contains the copy.
        """
        arr = NDTArray(self.dtype, **kwargs)
        return ext.copy(arr, self, **kwargs)

    def to_numpy(self):
        """Returns a NumPy array with the data contents and `dtype`.

        Returns
        -------
        numpy.array
            The NumPy array object containing the data of the whole Container.
        """
        return np.fromstring(self.to_buffer(), dtype=self.dtype).reshape(self.shape)
