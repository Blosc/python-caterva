from cat4py import container_ext as ext
import numpy as np

from .ndarray import NDArray, process_key


class NDTArray(NDArray):

    def __init__(self, dtype, **kwargs):
        """The multidimensional data container that plays well with NumPy.

        As this inherits from the :py:class:`Container` class, you can pass any
        keyword argument that is supported by the :py:meth:`Container.__init__`
        constructor, plus the `dtype`.

        Parameters
        ----------
        dtype: str
            The data type for the container.
        """
        if type(self) == NDTArray:
            self.pre_init(dtype, **kwargs)
        super(NDTArray, self).__init__(**self.kwargs)

    def pre_init(self, dtype, **kwargs):
        self.dtype = dtype
        kwargs["metalayers"] = {"type": {
            # TODO: adding "version" does not deserialize well
            # "version": 0,    # can be any number up to 127
            "dtype": self.dtype,
            }
        }
        self.kwargs = kwargs

    @classmethod
    def cast(cls, cont):
        cont.__class__ = cls
        assert isinstance(cont, NDTArray)
        return cont

    def __getitem__(self, key):
        """ Get a (multidimensional) slice as specified in key.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that step parameter is not honored yet in slices.

        Returns
        -------
        out: NDTArray
            An array, stored in a non-compressed buffer, with the requested data.
        """
        return self.slice(key)

    def slice(self, key, **kwargs):
        """ Get a (multidimensional) slice as specified in key. Generalizes :py:meth:`__getitem__`.

       Parameters
       ----------
       key: int, slice or sequence of slices
           The index for the slices to be updated. Note that step parameter is not honored yet in slices.

       Other Parameters
       ----------------
       kwargs: dict, optional
           Keyword arguments that are supported by the :py:meth:`cat4py.empty` constructor.

       Returns
       -------
       out: NDTArray
           An array with the requested data.
       """
        arr = NDTArray(self.dtype, **kwargs)
        return ext.get_slice(arr, self, process_key(key, self.ndim), **kwargs)

    @property
    def __array_interface__(self):
        interface = {
            "data": self,
            "shape": self.shape,
            "typestr": self.dtype,
            "version": 3
        }
        return interface

    def copy(self, **kwargs):
        """Copy into a new array.

        Other Parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments that are supported by the :py:meth:`cat4py.empty` constructor.

        Returns
        -------
        NDTArray
            An array containing the copy.
        """
        arr = NDTArray(self.dtype, **kwargs)
        return ext.copy(arr, self, **kwargs)