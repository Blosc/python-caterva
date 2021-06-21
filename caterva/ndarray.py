from . import caterva_ext as ext
import ndindex
from .info import InfoReporter
import os
from .meta import Meta


def process_key(key, shape):
    key = ndindex.ndindex(key).expand(shape).raw
    mask = tuple(True if isinstance(k, int) else False for k in key)
    key = tuple(k if isinstance(k, slice) else slice(k, k+1, None) for k in key)
    return key, mask


def prod(list):
    prod = 1
    for li in list:
        prod *= li
    return prod


def get_caterva_start_stop(ndim, key, shape):
    start = tuple(s.start if s.start is not None else 0 for s in key)
    stop = tuple(s.stop if s.stop is not None else sh for s, sh in zip(key, shape))

    size = prod([stop[i] - start[i] for i in range(ndim)])

    return start, stop, size


def parse_kwargs(**kwargs):
    if "urlpath" in kwargs and kwargs["urlpath"]:
        if os.path.exists(kwargs["urlpath"]):
            raise FileExistsError(f"Can not create the file {kwargs['urlpath']}."
                                  f"It already exists!")


class NDArray(ext.NDArray):
    def __init__(self, **kwargs):
        parse_kwargs(**kwargs)
        self.kwargs = kwargs
        super(NDArray, self).__init__(**self.kwargs)

    @classmethod
    def cast(cls, cont):
        cont.__class__ = cls
        assert isinstance(cont, NDArray)
        return cont

    @property
    def meta(self):
        if self.storage == "Blosc":
            return Meta(self)
        return None

    @property
    def info(self):
        """
        Print information about this array.
        """
        return InfoReporter(self)

    @property
    def info_items(self):
        items = []
        items += [("Type", f"{self.__class__.__name__} ({self.storage})")]
        items += [("Itemsize", self.itemsize)]
        items += [("Shape", self.shape)]

        if self.storage == "Blosc":
            items += [("Chunks", self.chunks)]
            items += [("Blocks", self.blocks)]
            items += [("Comp. codec", self.codec.name)]
            items += [("Comp. level", self.clevel)]
            filters = [f.name for f in self.filters if f.name != "NOFILTER"]
            items += [("Comp. filters", f"[{', '.join(map(str, filters))}]")]
            items += [("Comp. ratio", f"{self.cratio:.2f}")]
        return items

    @property
    def __array_interface__(self):
        interface = {
            "data": self,
            "shape": self.shape,
            "typestr": f"|S{self.itemsize}",
            "version": 3
        }
        return interface

    def __setitem__(self, key, value):
        key, mask = process_key(key, self.shape)
        start, stop, _ = get_caterva_start_stop(self.ndim, key, self.shape)
        key = (start, stop)
        return ext.set_slice(self, key, value)

    def __getitem__(self, key):
        """ Get a (multidimensional) slice as specified in key.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that step parameter is not honored yet
            in slices.

        Returns
        -------
        out: NDArray
            An array, stored in a non-compressed buffer, with the requested data.
        """
        return self.slice(key)

    def slice(self, key, **kwargs):
        """ Get a (multidimensional) slice as specified in key. Generalizes :py:meth:`__getitem__`.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that step parameter is not honored yet in
            slices.

        Other Parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments that are supported by the :py:meth:`caterva.empty` constructor.

        Returns
        -------
        out: NDArray
            An array with the requested data.
        """
        arr = NDArray(**kwargs)
        kwargs = arr.kwargs
        key, mask = process_key(key, self.shape)
        start, stop, _ = get_caterva_start_stop(self.ndim, key, self.shape)
        key = (start, stop)
        return ext.get_slice(arr, self, key, mask, **kwargs)

    def squeeze(self):
        """Remove the 1's in array's shape."""
        super(NDArray, self).squeeze(**self.kwargs)

    def to_buffer(self):
        """Returns a buffer with the data contents.

        Returns
        -------
        bytes
            The buffer containing the data of the whole array.
        """
        return super(NDArray, self).to_buffer(**self.kwargs)

    def copy(self, **kwargs):
        """Copy into a new array.

        Other Parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments that are supported by the :py:meth:`caterva.empty` constructor.

        Returns
        -------
        NDArray
            An array containing the copy.
        """
        arr = NDArray(**kwargs)
        return ext.copy(arr, self, **kwargs)
