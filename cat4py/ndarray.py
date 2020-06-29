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
        super(WriteIter, self).__init__(arr, **arr.kwargs)


class NDArray(ext.Container):
    def __init__(self, **kwargs):
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
        interface = {
            "data": self,
            "shape": self.shape,
            "typestr": f'S{self.itemsize}',
            "version": 3
        }
        return interface

    def __getitem__(self, key):
        """ Get a (multidimensional) slice as specified in key.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that step parameter is not honored yet in slices.

        Returns
        -------
        out: NDArray
            An array, stored in a non-compressed buffer, with the requested data.
        """
        return self.slice(key)

    def slice(self, key, **kwargs):
        """ Get a (multidimensional) slice as specified in key.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that step parameter is not honored yet in slices.
        kwargs: dict or None
            Keyword arguments that are supported by the :py:meth:`cat4py.empty` constructor.

        Returns
        -------
        out: NDArray
            An array with the requested data.
        """
        arr = NDArray(**kwargs)
        kwargs = arr.kwargs
        return ext.get_slice(arr, self, process_key(key, self.ndim), **kwargs)

    def iter_read(self, itershape=None):
        """Iterate over data blocks whose dims are specified in `itershape`.

        Parameters
        ----------
        itershape: tuple, list
            The shape in which the data block will be returned.  If `None`,
            the `NDArray.pshape` will be used as `itershape`.

        Yields
        ------
        out: tuple
            A tuple of (block, info)

            block: NDArray
                An array, stored in a non-compressed buffer, with the data block.
            info: namedtuple
                Info about the returned data block.  Its structure is:

                    slice: tuple
                        The coordinates where the data block starts.
                    shape: tuple
                        The shape of the actual data block (it can be
                        smaller than `itershape` at the edges of the array).
                    size: int
                        The size, in elements, of the block.
        """
        return ReadIter(self, itershape)

    def iter_write(self):
        """Iterate over non initialized data array.

        Yields
        ------
        out: tuple
            A tuple of (block, info)

            block: bytes
                The buffer with the data block to be filled.
            info: namedtuple
                Info about the data block to be filled.  Its structure is:

                    slice: tuple
                        The coordinates where the data block starts.
                    shape: tuple
                        The shape of the actual data block (it can be
                        smaller than `NDArray.blockshape` at the edges of the array).
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