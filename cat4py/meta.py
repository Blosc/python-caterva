from . import caterva_ext as ext
from collections.abc import Mapping


class Meta(Mapping):
    """
    Class providing access to user meta on a :py:class:`NDArray`.
    It will be available via the `.meta` property of an array.
    """
    def get(self, key, default=None):
        """Return the value for `key` if `key` is in the dictionary, else `default`.
        If `default` is not given, it defaults to ``None``"""
        return self[key] if key in self else default

    def __del__(self):
        pass

    def __init__(self, ndarray):
        self.arr = ndarray

    def __contains__(self, key):
        """Check if the `key` metalayer exists or not"""
        return ext.meta__contains__(self.arr, key)

    def __delitem__(self, key):
        return None

    def __setitem__(self, key, value):
        """Update the `key` metalayer with `value`.

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
        """Return the `item` metalayer.

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
        """Return the metalayers keys"""
        return ext.meta_keys(self.arr)

    def values(self):
        raise NotImplementedError("Values can not be accessed")

    def items(self):
        raise NotImplementedError("Items can not be accessed")

    def __iter__(self):
        """Iter over the keys of the metalayers"""
        return iter(self.keys())

    def __len__(self):
        """Return the nnumber of metalayers"""
        return ext.meta__len__(self.arr)
