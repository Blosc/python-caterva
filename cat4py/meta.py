from . import caterva_ext as ext
from collections.abc import MutableMapping


class Meta(MutableMapping):
    """
    Class providing access to user meta on a :py:class:`NDArray`.
    It will be available via the `.attrs` property of an array.
    """
    def __del__(self):
        pass

    def __init__(self, ndarray):
        self.arr = ndarray

    def __contains__(self, item):
        return ext.meta__contains__(self.arr, item)

    def __delitem__(self, key):
        return None

    def __setitem__(self, key, value):
        """Update the `name` metalayer with `content`.

        Parameters
        ----------
        key: str
            The name of the metalayer to update.
        value: bytes
            The buffer containing the new content for the metalayer.

            ..warning: Note that the *length* of the metalayer cannot not change,
            else an exception will be raised.
        """
        return ext.meta__setitem__(self.arr, key, value)

    def __getitem__(self, item):
        """Return the `name` metalayer.

        Parameters
        ----------
        item: str
            The name of the metalayer to return.

        Returns
        -------
        bytes
            The buffer containing the metalayer info (typically in msgpack
            format).
        """
        return ext.meta__getitem__(self.arr, item)

    def keys(self):
        return ext.meta_keys(self.arr)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return ext.meta__len__(self.arr)
