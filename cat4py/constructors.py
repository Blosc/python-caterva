from . import container_ext as ext
from .container import Container
from .tlarray import TLArray
from .nparray import NPArray


def empty(shape, dtype=None, **kwargs):
    """Create an empty container.

    In addition to regular arguments, you can pass any keyword argument that
    is supported by the :py:meth:`Container.__init__` constructor.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final container.
    dtype: numpy.dtype
        The dtype of the data.  Default: None.
    Returns
    -------
    TLArray or NPArray
        If dtype is None, a new :py:class:`Container` object is returned. If a
        dtype is passed, a new :py:class:`NPArray` is returned.
    """
    arr = TLArray(**kwargs) if dtype is None else NPArray(dtype, **kwargs)
    arr.updateshape(shape)
    return arr


def from_buffer(buffer, shape, dtype=None, **kwargs):
    """Create a container out of a buffer.

    In addition to regular arguments, you can pass any keyword argument that
    is supported by the :py:meth:`Container.__init__` constructor.

    Parameters
    ----------
    buffer: bytes
        The buffer of the data to populate the container.
    shape: tuple or list
        The shape for the final container.
     dtype: numpy.dtype
        The dtype of the data.  Default: None.

    Returns
    -------
    TLArray or NPArray
        If dtype is None, a new :py:class:`Container` object is returned. If a
        dtype is passed, a new :py:class:`NPArray` is returned.
    """
    arr = TLArray(**kwargs) if dtype is None else NPArray(dtype, **kwargs)
    ext.from_buffer(arr, shape, buffer)
    return arr


def from_numpy(nparray, dtype=None, **kwargs):
    """Create a container out of a NumPy array.

    In addition to regular arguments, you can pass any keyword argument that
    is supported by the :py:meth:`Container.__init__` constructor.

    Parameters
    ----------
    nparray: numpy.ndarray
        The NumPy array to populate the container with.
    dtype: numpy.dtype
        The dtype of the data.  Default: None.

    Returns
    -------
    TLArray or NPArray
        If dtype is None, a new :py:class:`Container` object is returned. If a
        dtype is passed, a new :py:class:`NPArray` is returned.
    """
    arr = from_buffer(bytes(nparray), nparray.shape, dtype=dtype,
                      itemsize=nparray.itemsize, **kwargs)
    return arr


def from_file(filename, copy=False):
    """Open a new container from `filename`.

    In addition to regular arguments, you can pass any keyword argument that
    is supported by the :py:meth:`Container.__init__` constructor.

    Parameters
    ----------
    filename: str
        The file having a Blosc2 frame format with a Caterva metalayer on it.
    copy: bool
        If true, the container is backed by a new, sparse in-memory super-chunk.
        Else, an on-disk, frame-backed one is created (i.e. no copies are made).

    Returns
    -------
    TLArray or NPArray
        If dtype is None, a new :py:class:`Container` object is returned. If a
        dtype is passed, a new :py:class:`NPArray` is returned.
    """

    arr = Container()
    ext.from_file(arr, filename, copy)
    if arr.has_metalayer("numpy"):
        arr.__class__ = NPArray()
        dtype = arr.get_metalayer("numpy")[b"dtype"]
        arr.pre_init(dtype)
    else:
        arr = TLArray.cast(arr)
        arr.pre_init()

    return arr


def from_sframe(sframe, copy=False):
    """Open a new container from `sframe`.

    Parameters
    ----------
    sframe: bytes
        The Blosc2 serialized frame with a Caterva metalayer on it.
    copy: bool
        If true, the container is backed by a new, sparse in-memory super-chunk.
        Else, an in-memory, frame-backed one is created (i.e. no copies are made).

    Returns
    -------
    Container
        The new :py:class:`Container` object.
    """
    arr = Container()
    ext.from_sframe(arr, sframe, copy)
    if arr.has_metalayer("numpy"):
        arr.__class__ = NPArray
        dtype = arr.get_metalayer("numpy")[b"dtype"]
        arr.pre_init(dtype)
    else:
        arr = TLArray.cast(arr)

        arr.pre_init()

    return arr
