# This module is only useful to been able to change the Container class to different subclasses.
# This can't be done on a Cython extension (ext.Container).


import msgpack
from . import container_ext as ext


def process_key(key, ndim):
    if not isinstance(key, (tuple, list)):
        key = (key,)
    key = tuple(k if isinstance(k, slice) else slice(k, k + 1) for k in key)
    if len(key) < ndim:
        key += tuple(slice(None) for _ in range(ndim - len(key)))
    return key


class Container(ext.Container):
    def __init__(self, **kwargs):
        """The low-level, multidimensional and type-less data container.

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

    def squeeze(self, **kwargs):
        """Remove the 1's in Container's shape."""
        super(Container, self).squeeze(**kwargs)

    def to_buffer(self, **kwargs):
        """Returns a buffer with the data contents.

        Returns
        -------
        bytes
            The buffer containing the data of the whole Container.
        """
        return super(Container, self).to_buffer(**kwargs)

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
        return super(Container, self).to_sframe(**kwargs)

    def copy(self, arr, **kwargs):
        """Copy into a new container.

        Returns
        -------
        Container
            The `arr` object containing the copy.
        """
        return super(Container, self).copy(arr, **kwargs)

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
        return super(Container, self).has_metalayer(name)

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
        content = super(Container, self).get_metalayer(name)

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
        return super(Container, self).update_metalayer(name, content)

    def get_usermeta(self):
        """Return the `usermeta` section.

        Returns
        -------
        bytes
            The buffer for the usermeta section (typically in msgpack format,
            but not necessarily).
        """
        content = super(Container, self).get_usermeta()
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
        return super(Container, self).update_usermeta(content)
